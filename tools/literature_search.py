"""Literature search tools: PubMed via NCBI E-utilities with demo fallback."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

USER_AGENT = "ScientificConsensusEngine/1.0 (Nucleate NYC BioHack 2026)"


def _http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """Search PubMed and return paper metadata with abstracts."""
    try:
        search_params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }
        )
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}"
        search_data = json.loads(_http_get(search_url))
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return _mock_papers(query, max_results)

        time.sleep(0.35)  # NCBI rate limit courtesy
        fetch_params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
        )
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{fetch_params}"
        xml_text = _http_get(fetch_url)
        return _parse_pubmed_xml(xml_text)
    except Exception:
        return _mock_papers(query, max_results)


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        journal_el = article.find(".//Title")
        year_el = article.find(".//PubDate/Year")

        pmid = pmid_el.text if pmid_el is not None else ""
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract = "".join(abstract_el.itertext()) if abstract_el is not None else ""
        journal = journal_el.text if journal_el is not None else ""
        year = year_el.text if year_el is not None else ""

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "source": "pubmed",
            }
        )
    return papers


def _mock_papers(query: str, max_results: int) -> list[dict]:
    """Demo fallback when PubMed is unavailable."""
    samples = [
        {
            "pmid": "35212345",
            "title": "TRAF2 signaling in CAR-T cell exhaustion and resistance",
            "abstract": (
                "TRAF2 mediates NF-kB activation downstream of CD28 costimulation. "
                "In CAR-T models, TRAF2 knockdown reduced IL-6 secretion and improved persistence."
            ),
            "journal": "Nature Immunology",
            "year": "2024",
            "source": "mock",
        },
        {
            "pmid": "35198765",
            "title": "IL-6 blockade reverses CAR-T dysfunction in solid tumors",
            "abstract": (
                "Elevated IL-6 correlates with CAR-T resistance. Tocilizumab co-therapy "
                "restored cytotoxicity in NF-kB-high tumor microenvironments."
            ),
            "journal": "Cancer Cell",
            "year": "2023",
            "source": "mock",
        },
        {
            "pmid": "34987654",
            "title": "NF-kB pathway inhibitors enhance adoptive cell therapy",
            "abstract": (
                "Pharmacologic NF-kB inhibition reduced exhaustion markers PD-1 and LAG-3 "
                "without compromising initial activation in preclinical models."
            ),
            "journal": "Science Translational Medicine",
            "year": "2023",
            "source": "mock",
        },
        {
            "pmid": "34876543",
            "title": "Contradictory role of TRAF2 in T cell homeostasis",
            "abstract": (
                "Complete TRAF2 loss causes lymphopenia and may limit therapeutic windows "
                "for CAR-T engineering strategies targeting this adaptor."
            ),
            "journal": "Journal of Experimental Medicine",
            "year": "2022",
            "source": "mock",
        },
        {
            "pmid": "34765432",
            "title": "Multi-omic atlas of CAR-T resistance mechanisms",
            "abstract": (
                "Integrative analysis identified antigen loss, checkpoint upregulation, "
                "and cytokine storm signatures as dominant resistance modes beyond TRAF2."
            ),
            "journal": "Cell Reports Medicine",
            "year": "2024",
            "source": "mock",
        },
    ]
    return samples[:max_results]


def search_literature(
    topic: str,
    sources: Optional[list[str]] = None,
    max_papers: int = 10,
) -> str:
    """Tool entry point: returns JSON string for function calling."""
    _ = sources  # arXiv/bioRxiv hooks reserved for future work
    papers = search_pubmed(topic, max_results=max_papers)
    return json.dumps({"topic": topic, "papers": papers, "count": len(papers)}, indent=2)


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": "Search PubMed for scientific papers on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Research topic or query"},
                "max_papers": {
                    "type": "integer",
                    "description": "Maximum number of papers to return",
                },
            },
            "required": ["topic"],
        },
    },
}
