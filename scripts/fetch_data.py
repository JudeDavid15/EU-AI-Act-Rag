import sys
from pathlib import Path
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger=logging.getLogger(__name__)


DOCUMENT={
    "eu_ai_act_2024_1689":{
        "celex":"32024R1689",
        "url":"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
        "format":"html"
    },
    "gdpr_2016_679":{
        "celex":"32016R0679",
        "url":"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        "format":"html"
    },

    "eu_ai_act_2024_1689_consolidated_20260727": {
        "celex": "02024R1689-20260727",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02024R1689-20260727",
        "format": "html"
    }

}

raw_data_dr=Path(__file__).resolve().parent.parent/"data"/"raw"

def fetch_and_save(doc_key:str,doc_info:dict)->Path:
    raw_data_dr.mkdir(parents=True,exist_ok=True)
    output_path=raw_data_dr/f"{doc_key}.{doc_info['format']}"

    if output_path.exists():
        logger.info(f"the file already exists {output_path}. Skipping Download")
        return output_path
    logger.info(f"Downloading {doc_key}(Celex:{doc_info['celex']}) from {doc_info['url']}")
    headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) EU-AI-Act-RAG-Research/1.0"}
    try:
        response=requests.get(doc_info["url"],headers=headers,timeout=30)
        response.raise_for_status()
        with open(output_path,'w',encoding='utf-8') as f:
            f.write(response.text)
        logger.info(f"Successfully saved to {output_path} ({len(response.text)} bytes)")
        return output_path
    except requests.RequestException as e:
        logger.error("Failed to download {doc_key} : {e}")
        sys.exit(1)

def main():
    logger.info("Starting raw data acquistion pipeline.....")
    for key, info in DOCUMENT.items():
        fetch_and_save(key,info)
    logger.info("Data Acquisition Complete")

if __name__=='__main__':
    main()
