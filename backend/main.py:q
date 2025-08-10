from typing import Tuple
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import hashlib
import json
import pickle
import redis
import os
from Clustering.association import AssociationAnalyzer
from RAG.query_data import query_rag
from Regression.Regression import PharmacySalesPredictor
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:5173"] in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
CACHE_TTL = 3600


def make_cache_key(func_name: str, params: dict):
    """Create a unique cache key from function name + parameters."""
    key_str = func_name + json.dumps(params, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache(key):
    data = r.get(key)
    if data:
        print("Redis Hit")
    else:
        print("Computed")
    return pickle.loads(data) if data else None


def set_cache(key, value, ttl=CACHE_TTL):
    r.setex(key, ttl, pickle.dumps(value))


DATA_FILE = "./assets/supertable.parquet"
A_Analyzer = AssociationAnalyzer()


@app.get("/association-rules")
async def association_rules_endpoint(
    brand_col: str = "brand_name",
    invoice_col: str = "invoice",
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    max_len: int = 2,
    top_n_brands: int = 200,
    min_brand_count: int = 20,
    sort_by: str = "lift",
    max_rules_display: int = 100,
    figure_width: int = 1400,
    figure_height: int = 800,
    return_html: bool = True,
    include_metrics_plot: bool = True,
):
    params = locals()
    cache_key = make_cache_key("association_rules", params)
    cached = get_cache(cache_key)
    if cached:
        return HTMLResponse(cached) if return_html else JSONResponse(cached)

    df = pd.read_parquet(DATA_FILE)
    result = A_Analyzer.association_rules(df=df, **params)

    set_cache(cache_key, result)
    return HTMLResponse(result) if return_html else JSONResponse(result)


@app.get("/association-graph")
async def association_graph_endpoint(
    ingredient_col: str = "ingredient_name",
    invoice_col: str = "invoice",
    min_product_count: int = 50,
    top_n_products: int = 160,
    cooccurrence_percentile: float = 75.0,
    lift_threshold: float = 1.1,
    max_connections: int = 150,
    clustering_method: str = "dbscan",
    dbscan_eps_range: Tuple[float, float] = (0.3, 1),
    dbscan_min_samples: int = 2,
    kmeans_k_range: Tuple[float, float] = (3, 12),
    max_clusters_display: int = 12,
    figure_width: int = 1500,
    figure_height: int = 1000,
    node_size_multiplier: float = 1.0,
    show_legend: bool = True,
    return_html: bool = True,
):
    params = locals()
    cache_key = make_cache_key("association_graph", params)
    cached = get_cache(cache_key)
    if cached:
        return HTMLResponse(cached) if return_html else JSONResponse(cached)

    df = pd.read_parquet(DATA_FILE)
    result = A_Analyzer.association_graph(df=df, **params)

    set_cache(cache_key, result)
    return HTMLResponse(result) if return_html else JSONResponse(result)


class QueryRequest(BaseModel):
    query: str


@app.post("/query-rag")
async def query_rag_endpoint(request: QueryRequest):
    response_text = query_rag(request.query)
    return {"query": request.query, "response": response_text}


@app.post("/predict-sales")
async def predict_sales_endpoint(config: dict = Body(...)):
    enhanced_config = {
        "data_file": DATA_FILE,
        "target_col": "sales_sheet",
        "sequence_length": 12,
        "forecast_periods": 30,
        "smoothing_span": 5,
        "lstm_units": 128,
        "lstm_epochs": 150,
        "dropout_rate": 0.3,
        "early_stopping": True,
        "confidence_level": 0.95,
        "theme": "dark",
        "remove_last_n": 1,
    }
    enhanced_config.update(config)

    cache_key = make_cache_key("predict_sales", enhanced_config)
    cached = get_cache(cache_key)
    if cached:
        return HTMLResponse(cached)

    predictor = PharmacySalesPredictor(config=enhanced_config)
    html_output = predictor.run_prediction()

    set_cache(cache_key, html_output)
    return HTMLResponse(html_output)


@app.post("/clear_cache")
def clear_cache():
    r.flushdb()
    return {"status": "success", "message": "Cache cleared"}


def main():
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
