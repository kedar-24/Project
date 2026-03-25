import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from services import parse_csv_file, get_preview_data, process_normalization

app = FastAPI(title="OmicsForge API", description="SOTA RNA-Seq Normalization Backend")

# Dynamic CORS configuration via environment variables for deployment flexibility
raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip().rstrip("/") for o in raw_origins.split(",") if o.strip()]

# Hardcoded fallbacks to ensure the specific deployment origin is always permitted
deployment_origin = "https://project-roan-six-31.vercel.app"
if deployment_origin not in allowed_origins:
    allowed_origins.append(deployment_origin)
if "http://localhost:3000" not in allowed_origins:
    allowed_origins.append("http://localhost:3000")

# If no restricted environment variable is set, allow all for debugging/testing
if not raw_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if "*" not in allowed_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": f"Internal Error: {str(exc)}"})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": f"Validation Error: {str(exc)}"})

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/")
def health_check():
    return {"status": "OmicsForge API is Online", "cors_mode": "Permissive" if "*" in allowed_origins else "Restricted"}

@app.post("/api/preview")
async def preview_csv(file: UploadFile = File(...)):
    """Receive file payloads, dispatch to Services, and return metadata properties."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format! Please upload a CSV.")
    
    try:
        content = await file.read()
        df = parse_csv_file(content)
        result = get_preview_data(df)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"detail": str(e)}
        )


@app.post("/api/normalize")
async def normalize_csv(
    file: UploadFile = File(...),
    gene_id_col: str = Form(...),
    do_tpm: str = Form("true"),
    do_rpkm: str = Form("true")
):
    """Receive files, execute computational transformations, and stream normalized datasets back."""
    try:
        is_tpm = do_tpm.lower() == "true"
        is_rpkm = do_rpkm.lower() == "true"
        
        content = await file.read()
        df = parse_csv_file(content)
        
        normalized_records = process_normalization(df, gene_id_col, is_tpm, is_rpkm)

        return JSONResponse(content={"result": normalized_records})
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"detail": str(e)}
        )

