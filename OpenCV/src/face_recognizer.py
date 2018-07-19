#导入OpenCV模块
import cv2
#导入os模块用于读取训练数据目录和路径
import os
# 导入numpy将python列表转换为numpy数组，OpenCV面部识别器需要它
import numpy as np

#我们的训练数据中没有标签0，因此索引/标签0的主题名称为空
subjects = ["", "messi", "cluo"]

#使用OpenCV用来检测脸部的函数
def detect_face(img):
    #将测试图像转换为灰度图像，因为opencv人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #加载OpenCV人脸检测器，我正在使用的是快速的LBP
    #还有一个更准确但缓慢的Haar分类器
    haar_path = '../cascade/lbp/lbpcascade_frontalface.xml'
    face_cascade = cv2.CascadeClassifier(haar_path)

    #让我们检测多尺度（一些图像可能比其他图像更接近相机）图像
    #结果是一张脸的列表
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #如果未检测到面部，则返回原始图像
    if (len(faces) == 0):
        return None, None
    
    #假设只有一张脸，
    #提取面部区域
    (x, y, w, h) = faces[0]
    
    #只返回图像的正面部分
    return gray[y:y+w, x:x+h], faces[0]

#该功能将读取所有人的训练图像，从每个图像检测人脸
#并将返回两个完全相同大小的列表，一个列表 
# 每张脸的脸部和另一列标签
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #获取数据文件夹中的目录（每个主题的一个目录）
    dirs = os.listdir(data_folder_path)
    
    #列表来保存所有主题的面孔
    faces = []
    #列表以保存所有主题的标签
    labels = []
    
    #让我们浏览每个目录并阅读其中的图像
    for dir_name in dirs:
        
        #我们的主题目录以字母's'开头
        #如果有的话，忽略任何不相关的目录
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #从dir_name中提取主题的标签号
        #目录名称格式= slabel
        #，所以从dir_name中删除字母''会给我们标签
        label = int(dir_name.replace("s", ""))
        
        #建立包含当前主题主题图像的目录路径
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #获取给定主题目录内的图像名称
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #浏览每个图片的名称，阅读图片，
        #检测脸部并将脸部添加到脸部列表
        for image_name in subject_images_names:
            
            #忽略.DS_Store之类的系统文件
            if image_name.startswith("."):
                continue;
            
            #建立图像路径
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #阅读图像
            image = cv2.imread(image_path)
            
#             #显示图像窗口以显示图像
#             cv2.imshow("Training on image...", image)
#             cv2.waitKey(100)
            
            #侦测脸部
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #为了本教程的目的
            #我们将忽略未检测到的脸部
            if face is not None:
                #将脸添加到脸部列表
                faces.append(face)
                #为这张脸添加标签
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


#让我们先准备好我们的训练数据
#数据将在两个相同大小的列表中
#一个列表将包含所有的面孔
#数据将在两个相同大小的列表中
print("Preparing data...")
faces, labels = prepare_training_data("../dataset/training_data")
print("Data prepared")

#打印总面部数和标签
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


# ----------------------- 面部检测完毕 -----------------------------------------------

# ----------------------- 下面是 面部识别 --------------------------------------------
#创建我们的LBPH人脸识别器
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#或者使用EigenFaceRecognizer替换上面的行
#face_recognizer = cv2.face.createEigenFaceRecognizer()

#或者使用FisherFaceRecognizer替换上面的行
#face_recognizer = cv2.face.createFisherFaceRecognizer()

#训练我们的面部识别器
face_recognizer.train(faces, np.array(labels))

#函数在图像上绘制矩形
#根据给定的（x，y）坐标和
#给定的宽度和高度
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#函数在从图像开始绘制文本
#通过（x，y）坐标。
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    
def predict(test_img):
    #制作图像的副本，因为我们不想更改原始图像
    img = test_img.copy()
    #从图像中检测脸部
    face, rect = detect_face(img)

    #使用我们的脸部识别器预测图像
    label = face_recognizer.predict(face)[0]
    #获取由人脸识别器返回的相应标签的名称
    label_text = subjects[label]
    
    #在检测到的脸部周围画一个矩形
    draw_rectangle(img, rect)
    #画预计人的名字
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Predicting images...")

# 加载测试图像
test_img1 = cv2.imread("../dataset/test_data/test1.jpg")
test_img2 = cv2.imread("../dataset/test_data/test2.png")

# print("lalla---",test_img1.shape)
# print("qqqqq---",test_img2.shape)

# # 加载测试图像
# test_img1 = "D:/opencvCas/test_data/test1.jpg"
# test_img2 = "D:/opencvCas/test_data/test2.png"

#执行预测
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

#显示两个图像
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()