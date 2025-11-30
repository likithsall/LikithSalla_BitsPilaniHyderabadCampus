import os
import json
import requests
import uvicorn
import tempfile
import time
import asyncio
import re
import traceback
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageEnhance

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        
    if not google_api_key:
        print("CRITICAL WARNING: GOOGLE_API_KEY not found. API calls will fail.")
    else:
        genai.configure(api_key=google_api_key)
        print("GenAI Configured successfully.")
except Exception as e:
    print(f"Warning: Failed to configure GenAI on startup: {e}")

# --- FASTAPI APP ---
app = FastAPI(
    title="HackRx Bill Extraction API (Final Fix)",
    description="High-precision extraction with Ghost-Item filtering."
)

# --- DEBUGGING HANDLER ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(f"❌ Validation Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": body.decode()},
    )

# --- DATA MODELS ---

class BillItem(BaseModel):
    item_name: str = Field(description="Exact name of the service/product/drug")
    item_amount: float = Field(description="The final Net Amount for the item")
    item_rate: Optional[float] = Field(description="The unit rate/price")
    item_quantity: Optional[float] = Field(description="The quantity count")

    # 1. Fill missing keys
    @model_validator(mode='before')
    @classmethod
    def fill_missing_keys(cls, data: Any):
        if isinstance(data, dict):
            if 'item_rate' not in data: data['item_rate'] = None
            if 'item_quantity' not in data: data['item_quantity'] = None
        return data

    # 2. Strict Float Cleaning
    @field_validator('item_amount', 'item_rate', 'item_quantity', mode='before')
    @classmethod
    def force_float(cls, v):
        if v is None: return None
        if isinstance(v, (float, int)): return float(v)
        if isinstance(v, str):
            clean_str = re.sub(r'[$,₹\s]', '', v)
            try: return float(clean_str)
            except ValueError: return 0.0
        return v

    # 3. Defaults
    @model_validator(mode='after')
    def set_defaults(self):
        if self.item_quantity is None:
            self.item_quantity = 1.0
        if self.item_rate is None:
            self.item_rate = 0.0
        return self

class PageEntry(BaseModel):
    page_no: Optional[str] = Field(description="The page number")
    page_type: Optional[str] = Field(description="Bill Detail | Final Bill | Pharmacy")
    bill_items: List[BillItem]

    # --- FIX: GHOST ITEM FILTER ---
    @model_validator(mode='after')
    def clean_bill_items(self):
        valid_items = []
        garbage_keywords = [
            "total", "subtotal", "balance", "carried", "brought", "page", 
            "continued", "gst", "cgst", "sgst", "discount", "amount", "rate", "qty"
        ]

        for item in self.bill_items:
            name_lower = item.item_name.lower().strip()
            
            # 1. Filter out empty names or super short names (likely noise)
            if len(name_lower) < 2:
                continue

            # 2. Filter out Headers/Totals masquerading as items
            if any(keyword in name_lower for keyword in garbage_keywords):
                continue

            # 3. Filter out rows that are just dates or numbers (Batch No hallucinations)
            # Checks if name is purely digits or date-like
            if re.match(r'^[\d\-\/\.]+$', name_lower):
                continue

            valid_items.append(item)
        
        self.bill_items = valid_items
        return self

    @model_validator(mode='before')
    @classmethod
    def fill_defaults(cls, data: Any):
        if isinstance(data, dict):
            if 'page_no' not in data or data.get('page_no') is None:
                data['page_no'] = "1"
            if 'page_type' not in data or data.get('page_type') is None:
                data['page_type'] = "Bill Detail"
        return data

    @field_validator('page_no', mode='before')
    @classmethod
    def clean_page_no(cls, v):
        if isinstance(v, str):
            match = re.search(r'\d+', v)
            if match: return match.group(0)
        return "1"

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageEntry]
    total_item_count: int

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

def preprocess_image(image_path: str) -> str:
    try:
        if not (image_path.lower().endswith(('.jpg', '.jpeg', '.png'))):
            return image_path
            
        print("🎨 Running Image Enhancement...")
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Higher contrast for Pharmacy bills
            enhancer = ImageEnhance.Contrast(img)
            contrast_img = enhancer.enhance(1.5) 
            
            # Sharpness helps with small text (Batch/Qty differentiation)
            enhancer_sharp = ImageEnhance.Sharpness(contrast_img)
            final_img = enhancer_sharp.enhance(1.2)

            processed_path = image_path + "_processed.png"
            final_img.save(processed_path)
            return processed_path
    except Exception as e:
        print(f"⚠️ Pre-processing failed: {e}")
        return image_path

def download_file(url_or_path: str) -> Tuple[str, str]:
    print(f"⬇️ Attempting to download: {url_or_path}")
    
    mime_type = "application/pdf"
    suffix = ".pdf"
    
    if url_or_path.startswith(('http://', 'https://')):
        try:
            response = requests.get(url_or_path, stream=True, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            url_lower = url_or_path.lower()
            
            if 'pdf' in content_type or '.pdf' in url_lower:
                mime_type = 'application/pdf'
                suffix = '.pdf'
            elif 'image' in content_type:
                mime_type = content_type
                suffix = '.png'
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
            
            print(f"✅ Downloaded to {path} ({mime_type})")
            return path, mime_type
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    else:
        if os.path.exists(url_or_path):
            if url_or_path.lower().endswith('.pdf'): return url_or_path, 'application/pdf'
            return url_or_path, 'image/jpeg'
        raise HTTPException(status_code=400, detail=f"Local file not found: {url_or_path}")

# --- CORE LOGIC ---

@app.post("/extract-bill-data", response_model=APIResponse, response_model_exclude_none=True)
async def extract_bill_data(request: APIRequest):
    temp_path = None
    processed_path = None
    upload_ref = None
    usage = TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)
    
    try:
        # 1. Download
        temp_path, mime_type = download_file(request.document)
        
        # 2. Pre-process (Image only)
        if "image" in mime_type:
            processed_path = preprocess_image(temp_path)
            if processed_path != temp_path:
                temp_path = processed_path
        
        # 3. Upload
        print(f"🚀 Uploading to Gemini...")
        upload_ref = await asyncio.to_thread(
            genai.upload_file, 
            path=temp_path, 
            mime_type=mime_type, 
            display_name="invoice_doc"
        )
        
        # 4. Wait
        print("⏳ Waiting for processing...")
        start_time = time.time()
        while upload_ref.state.name == "PROCESSING":
            if time.time() - start_time > 60: 
                raise TimeoutError("Gemini file processing timed out.")
            await asyncio.sleep(1)
            upload_ref = genai.get_file(upload_ref.name)
        
        if upload_ref.state.name == "FAILED":
             raise ValueError("Gemini failed to process the document.")

        # 5. Configure
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025", 
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ExtractionData,
                "temperature": 0.0
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # 6. Prompt (Updated to fix Ghost Items & Qty/Rate Confusion)
        prompt = """
        You are a high-precision financial data extractor.
        TASK: Extract individual transaction rows into JSON format.

        CRITICAL RULES:
        1. **Row Separation:** Extract strictly one JSON item per bill line.
        2. **Ignore Noise:** Do NOT extract Batch Numbers, Expiry Dates, or Headers as separate line items. If a line is just a date or a code, IGNORE it.
        3. **Column Discipline:** - 'Qty' column is distinct from 'Rate'. 
           - Do NOT confuse the 'Rate' value with 'Quantity'.
           - If Qty is not explicitly labeled, assume it is 1.
        4. **Exclusions:** IGNORE 'Total', 'Subtotal', 'Balance Due', 'Advance', or 'Discount' summary lines.
        5. **Formatting:** Output numbers with decimal points (e.g., 300.0).
        6. **Page Type:** Classify strictly as 'Bill Detail', 'Final Bill', or 'Pharmacy'.
        
        Output the data strictly in the requested JSON schema.
        """

        # 7. Generate
        print("🧠 Analyzing document...")
        response = await asyncio.to_thread(
            model.generate_content,
            [upload_ref, prompt]
        )
        
        # 8. Parse
        raw_data = json.loads(response.text)
        validated_data = ExtractionData(**raw_data)
        
        # Recalculate totals
        total_items = sum(len(p.bill_items) for p in validated_data.pagewise_line_items)
        validated_data.total_item_count = total_items

        # 9. Usage
        if response.usage_metadata:
            usage = TokenUsage(
                total_tokens=response.usage_metadata.total_token_count,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count
            )

        print("✅ Extraction Complete.")
        return APIResponse(is_success=True, token_usage=usage, data=validated_data)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        return APIResponse(is_success=False, token_usage=usage, data=None, error_message=str(e))
        
    finally:
        # Cleanup
        for path in [temp_path, processed_path]:
            if path and os.path.exists(path):
                try: os.remove(path)
                except: pass
        if upload_ref:
            try: await asyncio.to_thread(genai.delete_file, upload_ref.name)
            except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)