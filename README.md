# HackRx Bill Extraction API

## Description
This API extracts structured line-item data from medical bill documents using Google's Gemini 2.5 Flash AI model. It processes PDF or image documents to identify and extract detailed billing information including item names, amounts, rates, and quantities.

## Features
- FastAPI-based REST API
- Supports PDF and image documents (URLs or local files)
- Uses Google Gemini 2.5 Flash for accurate data extraction
- Extracts page-wise line items, page types (Bill Detail, Final Bill, Pharmacy)
- JSON output with token usage metrics

## Prerequisites
- Python 3.8+
- Google Generative AI API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Datathon_2911
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

1. Run the server:
   ```bash
   python main.py
   ```

   The API will start on http://localhost:8000

2. Send a POST request to `/extract-bill-data` with JSON payload:
   ```json
   {
       "document": "path/to/your/document.pdf"
   }
   ```
   or
   ```json
   {
       "document": "https://example.com/document.pdf"
   }
   ```

## API Reference

### POST /extract-bill-data

#### Request
- **Body**:
  ```json
  {
      "document": "string"  // URL or local file path
  }
  ```

#### Response
- **Success**:
  ```json
  {
      "is_success": true,
      "token_usage": {
          "total_tokens": 1234,
          "input_tokens": 567,
          "output_tokens": 667
      },
      "data": {
          "pagewise_line_items": [
              {
                  "page_no": "1",
                  "page_type": "Final Bill",
                  "bill_items": [
                      {
                          "item_name": "Aspirin",
                          "item_amount": 10.5,
                          "item_rate": 5.25,
                          "item_quantity": 2.0
                      }
                  ]
              }
          ],
          "total_item_count": 1
      }
  }
  ```

- **Error**:
  ```json
  {
      "is_success": false,
      "token_usage": {
          "total_tokens": 0,
          "input_tokens": 0,
          "output_tokens": 0
      },
      "data": null,
      "error_message": "Error description"
  }
  ```

## Testing

Run the test script:
```bash
python test_local.py
```

Make sure to update the `file_path` in `test_local.py` to point to a valid PDF document.

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Generative AI API key

## Notes
- The API supports reading documents from URLs or local file paths.
- Line items are extracted precisely, excluding totals and summary fields.
- Page types are classified as 'Bill Detail', 'Final Bill', or 'Pharmacy'.

This is a datathon (hackathon) project.
