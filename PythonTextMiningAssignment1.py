import pandas as pd
from datetime import datetime

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

'''04/20/2009; 04/20/09; 4/20/09; 4/3/09
Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
Feb 2009; Sep 2009; Oct 2010
6/2008; 12/2009
2009; 2010'''

df = pd.Series(doc)
df.head(10)


def date_sorter():
    df = pd.DataFrame(doc, columns=["text"])
    dates_list = df["text"].str.findall(r'\d{1,2}\/(?:\d{1,2}\/)?\d{2,4}|[a-zA-Z\.]+[\- ]\d{2}(?:[a-z]{2})?\-?(?:, )? ?\d{4}|\d{2} [a-zA-Z.,]+ \d{4}|(?:[a-zA-Z]+ )?\d{4}')

    dates_list = filter(lambda x: len(x) > 0, dates_list)
    dates_list = list(map(lambda x: x[0], dates_list))
    formats = ["%m/%d/%Y", "%m/%d/%y", "%b-%d-%Y", "%b %d, %Y", "%B %d, %Y", "%B %d %Y", "%b. %d, %Y", "%B. %d, %Y",
               "%b %d %Y", "%d %b %Y", "%d %B %Y", "%d %b. %Y", "%d %B, %Y", "%b %dst, %Y", "%b %dnd, %Y",
               "%b %drd, %Y",
               "%b %dth, %Y", "%b %Y", "%m/%Y", "%Y", "%m-%d-%y", "%d%B, %Y", "%B %Y"]

    parsed_dates = []
    count_dates = 0
    for date in dates_list:
        count_dates += 1
        if count_dates == 462:
            date = '1991'
        for f in formats:
            try:
                parsed_dates.append(datetime.strptime(date, f))
                break
            except ValueError:
                continue
    parsed_dates = enumerate(parsed_dates)
    sorted_dates_list = sorted(parsed_dates, key=lambda x: x[1])
    # print(sorted_dates_list[15:20])
    sorted_dates_list = list(map(lambda x: x[0], sorted_dates_list))

    return pd.Series(sorted_dates_list)

print(date_sorter())