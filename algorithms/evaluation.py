import os
import subprocess
import time
import numpy as np
import seaborn as sns


def getMojofm(index_X, predict, y_true):

    # basePath = '/home/sfy/Documents/VScodeProject/Thesis'
    basePath = "/home/sfy/Documents/VScodeProject/Thesis/result/"

    # write distra.rsf true label
    with open(basePath + os.sep + "distraTop11.rsf", "w") as f:
        for line in range(len(index_X)):
            f.write("contain " + str(y_true[line]) + " " + str(line) + "\n")

    # write distrb.rsf clustering result
    with open(basePath + os.sep + "distrbTop11.rsf", "w") as f:
        for line in range(len(index_X)):
            f.write("contain " + str(predict[line]) + " " + str(line) + "\n")

    time.sleep(1)

    # write to the command
    findPath = "cd /home/sfy/Documents/VScodeProject/Thesis"
    mojo = "java -cp /home/sfy/Documents/VScodeProject/Thesis/algorithms  mojo.MoJo /home/sfy/Documents/VScodeProject/Thesis/result/distraTop11.rsf /home/sfy/Documents/VScodeProject/Thesis/result/distrbTop11.rsf  -fm"

    subprocess.run(findPath, shell=True)
    time.sleep(0.5)
    subprocess.run(mojo, shell=True)
    """
    java -cp /home/sfy/Documents/VScodeProject/Thesis/algorithms  mojo.MoJo /home/sfy/Documents/VScodeProject/Thesis/result/distraTop11.rsf /home/sfy/Documents/VScodeProject/Thesis/result/distrbTop11.rsf  -fm
    # top 11 data
    # 80.41
    """


def get_similarity(y_true, y_predict, le, wholeList):
    # get jaccard
    # get the sets index of each label kmeans predicted
    # get the sets index of true label
    """

    true is 0 and label is 4
    0 -6 0.45
    1 -0 0.63
    4 -2 0.37
    5 -2 0.21
    7 -5 0.18
    8 -2 0.3
    9 -5 0.28

    for top 11
    0-4 0.36 !
    1-5 0.14
    2-1 0.13
    3-0 0.56 !
    4-8 0.99 !
    5-0 0.14
    6-1 0.22 ! 5 0.14
    7-5 0.48 !
    8-0 0.10
    9-1 0.25 -7 0.08  seven

    """
    pred_labels = []
    df = []
    for i in range(10):
        # index of each true
        b = np.where(y_predict == i)[0]
        similarity = list()
        for family in range(10):
            samples = np.where(y_true == family)[0]
            correctly_classified = list(set(samples).intersection(set(b)))
            union = list(set(samples).union(set(b)))
            jind = len(correctly_classified) / float(len(union))
            similarity.append(jind)

        lar = np.argmax(similarity)
        pred_label = "".join(le.inverse_transform([lar]))
        pred_labels.append(pred_label)
        print(
            f"--Result of cluster {i} the largest is {pred_label} with {similarity[lar]}"
        )

        # print(similarity)

        df.append(similarity)

    # make a pandas table for displain
    # DF = pd.DataFrame(df)
    # DF.to_csv('similarity.csv')

    # print out the not classfied labels
    print(set(wholeList) - set(pred_labels))


def plot_count(count_table):
    """
    return a table image in folder
    """

    table = count_table.to_frame("count")

    # sns.displot(table_count,x=table.index)
    # fam = table.index.to_series()
    # table = pd.concat([fam,table_count],axis=1,ignore_index=True)
    # table.reset_index()
    sns.set(rc={"figure.figsize": (30, 8.27)})
    sns.barplot(data=table, x=table.index, y=table["count"])
    # https://www.webucator.com/article/python-color-constants-module/
    # table.style.set_table_styles(
    #     [
    #         {
    #             'selector': 'tbody tr:nth-child(even)',
    #             'props': [('background-color', "#9AC0CD")]
    #         },
    #         {
    #             'selector': 'tbody tr:nth-child(odd)',
    #             'props': [('background-color', "#CDBE70")]
    #         }
    #     ]
    # )

    # import dataframe_image as dfi
    # dfi.export(
    #     table,
    #     "table.png"
    # )
