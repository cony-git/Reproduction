import csv
import os

def get_and_save_csv(txt_file_dir):
    with open("C:\\Users\\Cony\\Desktop\\corpus.txt", "r") as f_:
        corpus = f_.read().split()
    csv_f = open('C:\\Users\\Cony\\Desktop\\data.csv', 'a+', newline='')
    csv_writer = csv.writer(csv_f)
    # csv_writer.writerow(["Lables", "Strings"])
    for _, _, files in os.walk(txt_file_dir):
        i = 0
        for filename in files:
            filepath = txt_file_dir + filename
            i=i+1
            print(f"")
            with open(filepath, "r") as f:
                lines = ""
                words = f.read().split("\n")
                # print(words)
                for word in words:
                    if word in corpus:
                        lines = lines + word + " "
                print(lines)
                csv_writer.writerow(["1", lines])

    csv_f.close()

get_and_save_csv("E:\\NLP_STRING\\m_string\\")