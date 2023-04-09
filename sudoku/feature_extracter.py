import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from sudoku.trainer.model.backbone.resnet18 import ResNet_18
from sudoku.trainer.model.backbone.Lenet import CNN
import matplotlib.pyplot as plt


class FeatureExtracter:

    def __init__(self, image, config) -> None:
        self.image = image
        self.WIDTH = config["FeatureExtracter"]["Shape"]["width"]
        self.HEIGHT = config["FeatureExtracter"]["Shape"]["height"]
        self.model_dir = config["FeatureExtracter"]["model_dir"]
        model_name = config["Architecture"]["Backbone"]["name"]
        num_classes = config["Architecture"]["Backbone"]["num_classes"]
        assert model_name in ["ResNet18", "CNN"], "model not suppored!"
        model = {
            "ResNet18": ResNet_18,
            "CNN": CNN
        }
        self.model = model[model_name](3, num_classes=num_classes)
        self.probability_threshold = config["FeatureExtracter"]["probability_threshold"]
        self.size_sample = config["FeatureExtracter"]["size_image"]

    def preprocessing(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (15, 15), 1, 1)
        img_thersh = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
        return img_thersh

    def getBiggestContour(self, contours):
        biggest = np.array([[]])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                # Xấp xỉ contours bằng các đoạn thẳng
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    def reorder(self, myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

    def get_Perspective(self, img, masked_num, location, inv=False):
        """Takes original image as input"""
        pts1 = np.float32(location)
        pts2 = np.float32(
            [[0, 0], [self.WIDTH, 0], [0, self.HEIGHT], [self.WIDTH, self.HEIGHT]])

        width, height = self.WIDTH, self.HEIGHT
        if inv:
            pts1, pts2 = pts2, pts1
            width, height = img.shape[1], img.shape[0]

        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(masked_num, matrix, (width, height))
        return result

    def splitBoxes(self, img):
        rows = np.vsplit(img, 9)
        boxes = []
        for row in rows:
            cols = np.hsplit(row, 9)
            for box in cols:
                boxes.append(box)
        return boxes

    def transform(self, x):
        pil_img = Image.fromarray(np.uint8(x))

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size_sample),
            # torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            torchvision.transforms.ToTensor()
        ])

        return transform(pil_img).numpy()

    def getPredict(self, boxes):
        self.model.load_state_dict(torch.load(self.model_dir, map_location=torch.device('cpu')))
        imgs = torch.Tensor(np.array(list(map(self.transform, boxes))))
        y_hat = self.model(imgs)
        y_hat = torch.softmax(y_hat, dim=1)
        value, indices = torch.max(y_hat, dim=1)
        getClass = np.where(value < self.probability_threshold, 0, indices)
        return getClass.reshape(9, 9)

    def display_predict_number(self, num_pred, alpha=1, beta=0.5, gamma=1, color=(0, 255, 0)):
        img_empty = np.zeros(self.imgWarpColored.shape)
        for i in range(9):
            for j in range(9):
                if num_pred[i, j] != 0:
                    cv2.putText(img_empty, str(num_pred[i, j]), org=(j*50+10, i*50+35), fontScale=2, thickness=2,
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=color)

        inv = self.get_Perspective(
            self.img_resized, img_empty, self.biggest, inv=True)
        combined = cv2.addWeighted(
            inv, alpha, self.img_resized, beta, gamma, dtype=cv2.CV_8U)
        return np.clip(combined, 0, 255)

    def __call__(self):
        img_resized = cv2.resize(self.image, (self.HEIGHT, self.WIDTH))
        self.img_resized = img_resized
        thresh = self.preprocessing(img_resized)

        imgContours = img_resized.copy()
        imgBigContours = img_resized.copy()
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

        biggest, max_area = self.getBiggestContour(contours)

        if biggest.size != 0:
            biggest = self.reorder(biggest)
            cv2.drawContours(imgBigContours, biggest, -1, (0, 255, 0), 255)
            imgPerspective = self.get_Perspective(
                img_resized, img_resized, biggest)
            imgWarpColored = cv2.cvtColor(imgPerspective, cv2.COLOR_BGR2RGB)

        self.biggest = biggest
        self.imgWarpColored = imgWarpColored

        boxes = np.array(self.splitBoxes(imgWarpColored))
        pred = self.getPredict(boxes)
        return pred, boxes
