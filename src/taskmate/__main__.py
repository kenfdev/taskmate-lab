from dotenv import load_dotenv
import uvicorn


def main():
    load_dotenv()
    uvicorn.run(
        "taskmate.app:app",
        host="127.0.0.1",
        port=8765,
        proxy_headers=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
