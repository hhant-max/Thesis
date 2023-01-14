import pandas as pd
import warnings

warnings.simplefilter("ignore", UserWarning)

# read the whole file
filePath = "/home/sfy/Documents/VScodeProject/Thesis/data/negtive.csv"
DF = pd.read_csv(filePath)
name = DF["name"]

# create a new dataframe
data = pd.DataFrame()

familyList = [
    "Airpush", #6652
    "AndroRAT", # 46 less
    "Andup", # 43
    "Aples", # 20  
    "BankBot", #647
    "Bankun", # 70
    "Boqx", #215
    "Boxer", # 44 less
    "Cova",
    "Dowgin",
    "DroidKungFu",
    "Erop",
    "FakeAngry",  # 9210 less
    "FakeAV",  # 9220 less
    "FakeDoc",  # 9225
    "FakeInst",  # 9246
    "FakePlayer",  # 11414 less
    "FakeTimer",  # 11435 less
    "FakeUpdates",  # 11447 less
    "Finspy",  # 11452 less
    "Fjcon",  # 11461 less
    "Fobus",  # 11477 less
    "Fusob",  # 11481
    "GingerMaster",  # 12743 less
    "GoldDream",  # 12871 less
    "Gorpo",  # 12924 less
    "Gumen",  # 12947 less
    "Jisut",  # 13092 less
    "Kemoge",  # 13650 less
    "Koler",  # 13664 less
    "Ksapp",  # 13733 less
    "Kuguo",  # 13768
    "Kyview",  # 14966
    "Leech",  # 15141 less
    "Lnk",  # 15250 less
    "Lotoor",  # 15255
    "Mecor",  # 15584
    "Minimob",  # 17404
    "Mmarketpay",  # 17607 less
    "MobileTX",  # 17620 less
    "Mseg",  # 235
    "Mtk",  # 17872 less
    "Nandrobox",  # 17939 less(76)
    "Obad",  # 18015 less
    "Ogel",  # 18024 less
    "Opfake",  # 18030 less
    "Penetho",  # 18040 less
    "Ramnit",  # 18058 less
    "Roop",  # 18058 less
    "RuMMS",  # 402
    "SimpleLocker",  # 18516
    "SlemBunk",  # (174)
    "SmsKey",  # (165)
    "SmsZombie",  # 9 less
    "Spambot",  # 15 less
    "SpyBubble",  # 10 less
    "Stealer",  # 25 less
    "Steek",  # 12 less
    "Svpeng",  # 13 less
    "Tesbo",  # 5 less
    "Triada",  # 197
    "Univert", # 10 less
    "UpdtKiller", #24
    "Utchi", #12
    "Vidro", #23
    "VikingHorde", #7
    "Vmvol", #13
    "Winge", #19
    "Youmi", #1300
    "Zitmo", #24 less
    "Ztorg" #17 less
]
# 6654 6700 6743 6763 7410 7480 7695 7739 7756 8618 9164 9210


def addFam(data, family):
    # extract lines contating this family
    targetLines = DF[name.str.contains(f"({family})")]

    # add family name into it
    addFamil = [family for _ in range(len(targetLines))]
    targetLines.insert(1, "family", addFamil)

    res = pd.concat([data, targetLines], ignore_index=True)
    return res


for family in familyList:
    data = addFam(data, family)

data.to_csv("family.csv",index=False)

print(data.head(5))
