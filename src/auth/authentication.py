# src/auth/authentication.py
import os
from pathlib import Path
from typing import List, Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES: List[str] = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")
TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")


def _save_token(creds: Credentials, token_file: str) -> None:
    path = Path(token_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json(), encoding="utf-8")


def get_credentials(
    client_secrets_file: str = CLIENT_SECRETS_FILE,
    token_file: str = TOKEN_FILE,
    scopes: Optional[List[str]] = None,
    open_browser: bool = True,
    oauth_port: int = 0,  # fixed name
) -> Credentials:
    """
    Create/refresh Google OAuth2 credentials for Drive.
    Only handles auth; doesn't list/read any files.
    """
    scopes = scopes or SCOPES
    creds: Optional[Credentials] = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _save_token(creds, token_file)
        else:
            if not os.path.exists(client_secrets_file):
                raise FileNotFoundError(
                    f"Missing {client_secrets_file}. Put your client here or set GOOGLE_CLIENT_SECRETS."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            try:
                creds = flow.run_local_server(port=oauth_port) if open_browser else flow.run_console()
            except Exception:
                # Fallback if local server/browser fails (e.g., headless env)
                creds = flow.run_console()  # fixed
            _save_token(creds, token_file)

    return creds


def authenticate_drive(
    client_secrets_file: str = CLIENT_SECRETS_FILE,
    token_file: str = TOKEN_FILE,
    scopes: Optional[List[str]] = None,
    open_browser: bool = True,
    oauth_port: int = 0,  # fixed name
):
    """Return an authenticated Drive v3 service WITHOUT touching files."""
    creds = get_credentials(
        client_secrets_file=client_secrets_file,
        token_file=token_file,
        scopes=scopes or SCOPES,
        open_browser=open_browser,
        oauth_port=oauth_port,  # fixed
    )
    try:
        return build("drive", "v3", credentials=creds)
    except HttpError as e:
        raise RuntimeError(f"Failed to initialize Drive service: {e}") from e


def main():
    authenticate_drive()
    print("✅ Google Drive authentication completed. Service is ready.")


if __name__ == "__main__":
    main()







































# import os.path

# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError

# # If modifying these scopes, delete the file token.json.
# SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# def main():
#   """Shows basic usage of the Drive v3 API.
#   Prints the names and ids of the first 10 files the user has access to.
#   """
#   creds = None
#   # The file token.json stores the user's access and refresh tokens, and is
#   # created automatically when the authorization flow completes for the first
#   # time.
#   if os.path.exists("token.json"):
#     creds = Credentials.from_authorized_user_file("token.json", SCOPES)
#   # If there are no (valid) credentials available, let the user log in.
#   if not creds or not creds.valid:
#     if creds and creds.expired and creds.refresh_token:
#       creds.refresh(Request())
#     else:
#       flow = InstalledAppFlow.from_client_secrets_file(
#           "credentials.json", SCOPES
#       )
#       creds = flow.run_local_server(port=0)
#     # Save the credentials for the next run
#     with open("token.json", "w") as token:
#       token.write(creds.to_json())

#   try:
#     service = build("drive", "v3", credentials=creds)

#     # Call the Drive v3 API
#     results = (
#         service.files()
#         .list(pageSize=10, fields="nextPageToken, files(id, name)")
#         .execute()
#     )
#     items = results.get("files", [])

#     if not items:
#       print("No files found.")
#       return
#     print("Files:")
#     for item in items:
#       print(f"{item['name']} ({item['id']})")
#   except HttpError as error:
#     # TODO(developer) - Handle errors from drive API.
#     print(f"An error occurred: {error}")


# if __name__ == "__main__":
#   main()

# new version