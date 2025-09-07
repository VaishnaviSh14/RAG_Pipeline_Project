def main():
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Hello from rag-pipeline! (info log)")
    logger.error("This is a test error log from main.py")
    print("Hello from rag-pipeline!")


if __name__ == "__main__":
    main()
