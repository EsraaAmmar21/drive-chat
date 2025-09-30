# # download related pdfs for user query

# from io import BytesIO
# from typing import Optional
# from googleapiclient.discovery import Resource
# from googleapiclient.http import MediaIoBaseDownload
# from pypdf import PdfReader


# def download_pdf_bytes(service:Resource, file_id:str)->bytes:
#     """
#     Download a pdf from google drive and return its raw bytes
#     """
#     request = service.files().get_media(file_Id =file_id)
#     buf = BytesIO()
#     downloader = MediaIoBaseDownload(buf,request)
#     done = False
#     while not done :
#         _,done = downloader.next_chunk()
#     return buf.getvalue()


# def extract_pdf_text(pdf_bytes:bytes ,max_pages:Optional[int]= None)->str:
#     """
#     Extract text from PDF bytes using pypdf.
#     If max_page is provided , only read the first N pages(fast peeking)
#     Non-text pages are skipped; we do NOt OCR by design

#     """
#     reader = PdfReader(BytesIO(pdf_bytes))
#     n_pages = len(reader.pages)
#     end = n_pages  if max_pages is None else min(max_pages , n_pages)
#     out =[]
#     for i in range(end):
#         try:
#             out.append(reader.pages[i].extract_text() or "")
#         except Exception:
#             continue
#     return "\n".join(out).strip()


from io import BytesIO
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseDownload

def download_pdf_bytes(service: Resource, file_id: str) -> bytes:
    """
    Download a PDF from Google Drive and return its raw bytes.
    No parsing here; higher layers decide how to parse.
    """
    request = service.files().get_media(fileId=file_id)
    buf = BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

