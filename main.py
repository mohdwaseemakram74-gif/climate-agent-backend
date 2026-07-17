from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_research_agent

app = FastAPI(title="Climate Action Research Agent API")

# Allow your Vercel frontend to call this backend.
# Replace "*" with your actual Vercel URL once deployed, e.g.
# ["https://your-app.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str


class ResearchResponse(BaseModel):
    topic: str
    report: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Climate Action Agent API is running"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    try:
        report = run_research_agent(req.topic)
        return ResearchResponse(topic=req.topic, report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
