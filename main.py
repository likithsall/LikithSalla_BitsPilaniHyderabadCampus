import os
import json
import requests
import uvicorn
import tempfile
import time
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---

# Configure GenAI immediately
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    # If using Render, os.getenv works automatically with Environment Variables set in Dashboard
    if not google_api_key:
        # Fallback for local testing if .env is missing but variable is set in system
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        
    if not google_api_key:
        print("CRITICAL WARNING: GOOGLE_API_KEY not found in environment.")
    else:
        genai.configure(api_key=google_api_key)
        print("GenAI Configured successfully.")
except Exception as e:
    print(f"Warning: Failed to configure GenAI on startup: {e}.")

# 2. FASTAPI APPLICATION DEFINITION
app = FastAPI(
    title="HackRx Bill Extraction API",
    description="API for extracting structured line-item data from bill documents using Gemini 2.5 Flash."
)

# --- DATA MODELS ---

class BillItem(BaseModel):
    item_name: str = Field(description="Name of the service/medicine exactly as in bill.")
    item_amount: float = Field(description="Net amount of the item post-discounts.")
    item_rate: float = Field(description="Unit rate of the item.")
    item_quantity: float = Field(description="Quantity of the item.")

class PageEntry(BaseModel):
    # Made Optional to prevent validation errors if AI misses it
    page_no: Optional[str] = Field(description="The page number.") 
    page_type: str = Field(description="'Bill Detail', 'Final Bill', or 'Pharmacy'")
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageEntry]
    total_item_count: Optional[int] = Field(description="Total count of items.")

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error_message: Optional[str] = None

class APIRequest(BaseModel):
    document: str

# --- HELPER FUNCTIONS ---

def download_file(url_or_path: str) -> Tuple[str, str]:
    """
    Downloads file and returns (local_path, mime_type).
    """
    print(f"Attempting to process: {url_or_path}")
    
    # Default fallback
    mime_type = "application/pdf"
    suffix = ".pdf"
    
    if url_or_path.startswith(('http://', 'https://')):
        try:
            response = requests.get(url_or_path, stream=True, timeout=30)
            response.raise_for_status()
            
            # 1. Try to get type from Header
            content_type = response.headers.get('Content-Type', '').lower()
            
            # 2. If Header is generic/empty, check URL extension logic
            if 'pdf' in content_type:
                mime_type = 'application/pdf'
                suffix = '.pdf'
            elif 'image' in content_type:
                mime_type = content_type
                suffix = '.png' # default suffix for images
                if 'jpeg' in content_type or 'jpg' in content_type: suffix = '.jpg'
            else:
                # Fallback: Check URL string for extension (handles Azure Blob URLs)
                url_lower = url_or_path.lower()
                if '.pdf' in url_lower:
                    mime_type = 'application/pdf'
                    suffix = '.pdf'
                elif '.png' in url_lower:
                    mime_type = 'image/png'
                    suffix = '.png'
                elif '.jpg' in url_lower or '.jpeg' in url_lower:
                    mime_type = 'image/jpeg'
                    suffix = '.jpg'

            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, 'wb') as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
            print(f"Downloaded to {path} with inferred mime_type: {mime_type}")
            return path, mime_type
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    else:
        # Local file handling
        if os.path.exists(url_or_path):
            if url_or_path.lower().endswith('.pdf'):
                return url_or_path, 'application/pdf'
            elif url_or_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                return url_or_path, 'image/jpeg'
            return url_or_path, 'application/pdf'
        raise HTTPException(status_code=400, detail=f"Local file not found: {url_or_path}")

# --- CORE LOGIC ---

@app.post("/extract-bill-data", response_model=APIResponse, response_model_exclude_none=True)
async def extract_bill_data(request: APIRequest):
    temp_path = None
    upload_ref = None
    usage = TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)
    
    try:
        # 1. Download (Returns tuple: path AND mime_type)
        temp_path, mime_type = download_file(request.document)
        
        # 2. Upload to Gemini
        print(f"Uploading file with mime_type: {mime_type}...")
        try:
            # FIX: Explicitly pass mime_type to Gemini
            upload_ref = genai.upload_file(path=temp_path, mime_type=mime_type, display_name="medical_bill_document")
        except Exception as e:
            raise ValueError(f"Upload failed. Check API Key. Error: {e}")
        
        # Wait for processing
        while upload_ref.state.name == "PROCESSING":
            time.sleep(1)
            upload_ref = genai.get_file(upload_ref.name)
        
        if upload_ref.state.name == "FAILED":
             raise ValueError("GenAI processing failed.")

        # 3. Model Config
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025", 
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ExtractionData,
                "temperature": 0.0
            }
        )

        # 4. Prompt
        prompt = """
        You are an expert bill data extractor. Analyze the uploaded document(s) page by page. 
        
        **CRITICAL INSTRUCTIONS for high accuracy:**
        1. **Line Items Only:** Extract every single service, drug, or consumable item that has a quantity and rate.
        2. **Exclude Totals:** IGNORE subtotals, 'Total Amount', 'Balance Due', or any summary lines. Only extract specific line item entries.
        3. **Double Check:** Ensure that you do not double count any line item entries.
        4. **Data Types:** Ensure all amount, rate, and quantity values are correctly parsed as floats.
        5. **Item Name:** Must be extracted exactly as mentioned in the bill.
        6. **Page Number:** If the page number is not explicitly visible, assume it is "1".
        7. **Page Type:** Classify the page content as one of 'Bill Detail', 'Final Bill', or 'Pharmacy'.
        
        Output the data strictly in the requested JSON schema.
        """

        # 5. Generate
        print("Generating content...")
        response = model.generate_content([upload_ref, prompt])
        
        # 6. Parse
        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        parsed_data = json.loads(result_text)
        
        # --- POST-PROCESSING ---
        pagewise = parsed_data.get('pagewise_line_items', [])
        if pagewise:
            for page in pagewise:
                if not page.get('page_no'):
                    page['page_no'] = "1"
        
        # 7. Recalculate Totals
        total_count = sum(len(p.get('bill_items', [])) for p in pagewise)
        parsed_data['total_item_count'] = total_count

        # 8. Usage
        if response.usage_metadata:
            usage = TokenUsage(
                total_tokens=response.usage_metadata.total_token_count,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count
            )

        return APIResponse(is_success=True, token_usage=usage, data=parsed_data)

    except Exception as e:
        print(f"ERROR CAUGHT: {e}")
        return APIResponse(is_success=False, token_usage=usage, data=None, error_message=str(e))
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if upload_ref:
            try:
                genai.delete_file(upload_ref.name)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)