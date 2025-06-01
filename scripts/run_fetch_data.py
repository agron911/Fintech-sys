from src.crawler.yahoo_finance import delete_files, crawl_all_ch, crawl_otc_yf

if __name__ == "__main__":
    print("Deleting old files...")
    delete_files()
    print("Crawling all CH stocks...")
    crawl_all_ch()
    print("Crawling OTC stocks...")
    crawl_otc_yf()
    print("Done.") 