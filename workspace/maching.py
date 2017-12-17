import cv2
import os


class somayaki:
    def __init__(self):
        pass

    def match(self, filepath):
        TARGET_FILE = "./somayaki/test/" + filepath
        print("taeget file : {0}".format(filepath))
        IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/somayaki/matchpic/'

        IMG_SIZE = (200, 200)

        target_img_path = TARGET_FILE
        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, IMG_SIZE)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.ORB_create()
        #detector = cv2.AKAZE_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)

        files = os.listdir(IMG_DIR)

        for file in files:
            
            EV_dic = {}
            
            if file == '.DS_Store' or file == TARGET_FILE:
                continue

            comparing_img_path = IMG_DIR + file

            try:
                comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)

                comparing_img = cv2.resize(comparing_img, IMG_SIZE)

                (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)

                matches = bf.match(target_des, comparing_des)

                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)

            except cv2.error:
                ret = 100000

            print(file, ret)
            EV_dic[file] = ret
            
#            output = sorted(EV_dic.items(), key=lambda x: x[1])
#            print(output[-1])
#            print("Very similar to " + str(output[-1][0]))

        output = sorted(EV_dic.items(), key=lambda x: x[1])
        print(output[-1])
        print("Very similar to " + str(output[-1][0])) 
        print("")


if __name__ == '__main__':
    Somayaki = somayaki()
    DIR = os.path.abspath(os.path.dirname(__file__)) + '/somayaki/test/'
    print(DIR)
    f = os.listdir(DIR)

    fiule_ev = os.listdir(DIR)

    print(fiule_ev)
    for testfile in fiule_ev:
        Somayaki.match(testfile)

