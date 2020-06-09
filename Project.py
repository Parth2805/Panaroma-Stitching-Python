import numpy as np
import cv2
import random
random.seed(1)


#--------------------------------------Assignment-2----------------------------------------------------------
def detectCorner(name):
    
    image = cv2.imread(name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows = image.shape[0]
    cols = image.shape[1]
    
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    Ixx = cv2.GaussianBlur(dx*dx,(3,3),0,borderType= cv2.BORDER_DEFAULT)
    Ixy = cv2.GaussianBlur(dx*dy,(3,3),0,borderType= cv2.BORDER_DEFAULT)
    Iyy = cv2.GaussianBlur(dy*dy,(3,3),0,borderType= cv2.BORDER_DEFAULT)

    c=np.zeros_like(gray,dtype=np.float32)
    angle = np.arctan2(dy,dx)*180/np.pi
    magnitude = np.sqrt((dx**2)+(dy**2))
    trace = Ixx + Iyy + 1e-8
    det = Ixx*Iyy - Ixy*Ixy
    
    r = det/trace
    
    for y in range(0,rows):
        for x in range(0,cols):
            
#            print(r[y,x])
            if r[y,x]>23000:
                c[y,x]=r[y,x]
    
#    cv2.imshow("1a",c)
#    cv2.imwrite("1a.png",c)
    
#    finalcorner = nonmaxSuppression(c,angle,image)
#    return finalcorner
    return c,angle,magnitude
 
def nonmaxSuppression(c,angle,image):
    
    threshold = 35000
    features=[]
    offset = 1
    nonMax = np.zeros_like(c,dtype = np.float32)
    for y in range(offset,c.shape[0] - offset):
        for x in range(offset, c.shape[1] - offset):
            
            neigh = c[y-offset:y-offset+3, x-offset:x-offset+3]
            
            min_,max_,minpoint_,maxpoint_ = cv2.minMaxLoc(neigh)
            
            temp = np.zeros((3,3), dtype = np.float32)
            
            if max_ > threshold:
                
                temp[maxpoint_[1],maxpoint_[0]]= max_
                nonMax[y-offset:y-offset+3, x-offset:x-offset+3]=temp
                
                fp = cv2.KeyPoint()
                fp.angle=angle[y,x]
                fp.size=10
                fp.response=c[y,x]
                fp.pt=(x,y)
                features.append(fp)
     
    print("Features:",len(features))           
    final=cv2.drawKeypoints(image,features,None,color=(0,255,0),flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
#    
#    cv2.imshow("final",final)
#    cv2.waitKey(0)
    temp_features = []
    
    for point in features:
        
        radmin=-1
        mx,my=point.pt
        mx,my=int(mx),int(my)
        for point_ref in features:
            
            rx,ry = point_ref.pt
            rx,ry = int(rx) , int (ry)
            if point.pt!=point_ref.pt :
                
                if c[my,mx] < 0.8*c[ry,rx]:
                    
                    distance = float(np.sqrt((mx-rx)**2+(my-ry)**2))
#                    print (distance)
                    if radmin==-1:
                        radmin = distance
                    
                    if distance < radmin:
                        radmin = distance
        if radmin == -1 :
            radmin = 10000000
            
#        radiusDistance[y,x]=radmin
        temp_features.append((point,radmin))            
            
            
    
    temp_features.sort(key = lambda x: x[1],reverse=True)
    best_features=[]
    
    for feature in temp_features:
        best_features.append(feature[0])
        
                  
    final2=cv2.drawKeypoints(image,best_features[:300],None,color=(0,255,0),flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
#    cv2.imshow("Non-Max Suppresion",np.concatenate((final,final2),axis=1))
#    cv2.imwrite("1a.png",final2)
    cv2.waitKey(0)   
    return best_features , nonMax           

def featureDescriptor(features,image,angle,magnitude):
    
    
    histogram=[]
    cols = image.shape[1]
    rows = image.shape[0]
    for feature in features:
        
        x = int(feature.pt[0])
        y = int(feature.pt[1])
        
        if (x-8)>=0 and x+8<=cols and (y-8)>=0 and y+8 <=rows:
        
            temp_magnitude = magnitude[y-8:y+8,x-8:x+8]
            temp_angle = angle[y-8:y+8, x - 8: x + 8]
                        
            temp_var_angle= angleInv(temp_angle)
            
            norm_mag = cv2.normalize(temp_magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            
            list_ =[]
            for i in range(0,16,4):
                
                for j in range(0,16,4):
                        
#                    print(i,j)
                    mag = norm_mag[i:i+4,j:j+4]
                    ang = temp_var_angle[i:i+4,j:j+4]
                    temp_hist = calhistogram(mag,ang)
                    
                    for value in temp_hist:
                        
                        list_.append(value)
                                   
            histogram.append(list_)           
                       
                    
    return np.asarray(histogram)                   



def findMatches(descriptor1,descriptor2):
    
    index1=0
    matches=[]
    ssd = []
    for point1 in descriptor1:
    
        
        sum_of_square = (descriptor2-point1)**2
        sum_of_square = np.sum(sum_of_square,axis = 1)
        min_index = np.argmin(sum_of_square)
        
        
        temp_sum = sum_of_square.copy()
        temp_sum = np.delete(temp_sum,min_index)
        
        min_index_second = np.argmin(temp_sum)
        min_value_second = temp_sum[min_index_second]
        
        min_value = sum_of_square[min_index]
        

        ratio= float(min_value/min_value_second)
#        vector.append(min_index_second)
        
        index2=min_index
       
        if(ratio < 0.8):
#            print(min_value_second, min_value_second)
#            print(ratio)
            match = cv2.DMatch(index1,index2,ratio)
            matches.append(match)
            ssd.append(ratio)
                
        index1 =index1 + 1
        matches = sorted(matches, key = lambda x:x.distance)
    return matches  , ssd  
    
    
                    
def Sort(list_):  
   
    list_.sort(key = lambda x: x[1],reverse=True)  
    return list_              
                
def angleInv(angle_):
    
    temp_angle = angle_.copy()
    
    
    
    temp_hista=np.zeros(36,dtype = int)
    for y in range(angle_.shape[0]):
        for x in range(angle_.shape[1]):
            
            
            t_a = temp_angle[y,x]
#            print("TA:",t_a)
            if(t_a<0):
                t_a = t_a + 360
#                
            
            index = int((t_a / 10))
#           
            temp_hista[index] = temp_hista[index] + 1
    
#    dominant_orientation = np.amax(temp_hista)
#    result = np.where(temp_hista == np.amax(temp_hista))
    result = np.argmax(temp_hista)
#    minvalue,maxValue, minValueIndex , dominant_index = cv2.minMaxLoc(temp_angle)
    
    
#    print("D:",dominant_orientation)    
    
    
    for y in range(angle_.shape[0]):
        for x in range(angle_.shape[1]):
              
            temp_angle[y,x]=temp_angle[y,x]-(result*10)

    return temp_angle


def calhistogram(mag,ang):
    
    hist = np.zeros(8 ,dtype=np.float32)
    
    for y in range(ang.shape[0]):
        for x in range(ang.shape[1]):
            
            a = ang[y,x]
#            
#            while(a>360):
#                a = a - 360
#            while(a<-360):
#                a = a + 360
#            print("Before a:",a)
            if(a<0):
                a = a + 360
#                print("After a:",a)
            
            temp_a = int((a / 45))
#            print("Temp_a:",temp_a)
            hist[temp_a] = hist[temp_a] + mag[y,x]
    
    hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    for i in range(8):
        if(hist[i]>0.2):
            hist[i]=0.2
                    
    return hist

#-------------------------------------------------------------Project-------------------------------------------
def project(x1,y1,H1):

      
    px2 = (H1[0,0]*x1+H1[0,1]*y1+H1[0,2])/(H1[2,0]*x1+H1[2,1]*y1+H1[2,2])
    
    py2 = (H1[1,0]*x1+H1[1,1]*y1+H1[1,2])/(H1[2,0]*x1+H1[2,1]*y1+H1[2,2])
    

    return px2 ,py2


def computeInlierCount(H, matches_ ,keypoint1 ,keypoint2, inlierThreshold=50):
    
    numofinlier = 0 
    for match in matches_:
        
        x1, y1 = np.float32(keypoint1[match.queryIdx].pt)
        x2, y2 = np.float32(keypoint2[match.trainIdx].pt)

        proj_X ,proj_Y = project(x1,y1,H)   
               
        distance = np.sqrt((proj_X-x2)**2 + (proj_Y-y2)**2)

        if(distance < inlierThreshold):       
            numofinlier = numofinlier + 1

    return numofinlier

def computeInliers(H, matches_ ,keypoint1 ,keypoint2, inlierThreshold=20):
     
    inliers = [] 
    for match in matches_:
        
        x1, y1 = np.float32(keypoint1[match.queryIdx].pt)
        x2, y2 = np.float32(keypoint2[match.trainIdx].pt)
        
        proj_X ,proj_Y = project(x1,y1,H)   
        
        distance = np.sqrt((proj_X-x2)**2 + (proj_Y-y2)**2)
        if(distance < inlierThreshold):
            
           inliers.append(match)

    return np.asarray(inliers)

def RANSAC(matches,numIterations,keypoint1, keypoint2):
    
    best_inliers = 0
    for i in range(numIterations):
        
        random_matches = random.sample(matches, k=4)
       
        src = np.float32([keypoint1[match.queryIdx].pt for match in random_matches])
        dest = np.float32([keypoint2[match.trainIdx].pt for match in random_matches])
        
        temp_H, status = cv2.findHomography(src,dest,0)
        inlierscount = computeInlierCount(temp_H,matches,keypoint1, keypoint2)
        
        if inlierscount > best_inliers :
            best_inliers = inlierscount
            best_H = temp_H
        
    final_inliers = computeInliers(best_H, matches, keypoint1, keypoint2)
    
    final_H , status = cv2.findHomography(np.float32([keypoint1[match.queryIdx].pt for match in final_inliers]),np.float32([keypoint2[match.trainIdx].pt for match in final_inliers]),0)

    return final_inliers, final_H, np.linalg.inv(final_H)

def stitch(image1, image2, H, HomInv):

    tl = (0,0)
    tr = (0,image2.shape[0])
    bl = (image2.shape[1], 0)
    br = (image2.shape[1], image2.shape[0])
    corners = np.array([[tl],
                        [tr],
                        [bl],
                        [br]])
      

    image1corners = np.array([[0,0],[0,image1.shape[0]],[image1.shape[1],image1.shape[0]],[image1.shape[1],0]])

    stitched_corners = np.array([project(corner[:,0],corner[:,1],HomInv) for corner in corners],dtype=np.int32).reshape(-1,2)

    corners = corners.reshape(-1,2)
   
    minx = np.minimum(np.min(image1corners[:,0]),np.min(stitched_corners[:,0]))
    miny = np.minimum(np.min(image1corners[:,1]),np.min(stitched_corners[:,1]))  
    maxx = np.maximum(np.max(image1corners[:,0]),np.max(stitched_corners[:,0]))  
    maxy = np.maximum(np.max(image1corners[:,1]),np.max(stitched_corners[:,1]))
       
    xh = HomInv
    a = image1  
    b = image2          

    homt = np.array([[1,0,-minx],[0,1,-miny],[0,0,1]])
    input_h = np.dot(homt,xh)
    offsetx = maxx - minx
    offsety = maxy - miny

    tmp = cv2.warpPerspective(b, input_h, (offsetx,offsety))

    
#    cv2.imshow("Warped Perspective", tmp)
    x,y=0,0
    for i in range(-minx,-minx+a.shape[1]):
        y=0
        for j in range(-miny,-miny+a.shape[0]):
             
            tmp[j,i] = a[y,x] if all(tmp[j,i])==0 else tmp[j,i] 
            y=y+1
        x=x+1
#    cv2.imshow("StitchedImage", tmp)
    cv2.waitKey(0)
    return tmp
    
    

def main(first,second):
    name1 = "Rainier"+str(first)+".png"
    img1 = cv2.imread(name1,cv2.COLOR_BGR2GRAY)  
    name2 = "Rainier"+str(second)+".png"      
    img2 = cv2.imread(name2,cv2.COLOR_BGR2GRAY)  

    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
        
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches_refined, hom, hom_inv = RANSAC(matches,100, kp1, kp2)
            
    finalimage = stitch(img1, img2, hom, hom_inv)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None)
    img4 = cv2.drawMatches(img1,kp1,img2,kp2,matches_refined,None)
#    cv2.imshow("Final",img3)
#    cv2.imshow("Final2",img4)
    
    if first == 1:
        
        image_first = img1.copy()
        image_second = img2.copy()
        c1, angle1 , magnitude1 = detectCorner(name1)
        c2, angle2 , magnitude2 = detectCorner(name2)
        
        features1, nonMaxSupp1 = nonmaxSuppression(c1,angle1,image_first)
        features2,nonMaxSupp2 = nonmaxSuppression(c2,angle2,image_second)
        
        #_ , descriptor_1 = sift.detectAndCompute(image_first,None)
        #_ , descriptor_2 = sift.detectAndCompute(image_second,None)
        
        descriptor_1 = featureDescriptor(features1,image_first,angle1,magnitude1)
        descriptor_2 = featureDescriptor(features2,image_second,angle2,magnitude2)
        
        matches, ssdRatio = findMatches(descriptor_1,descriptor_2)
        matches_refined, hom, hom_inv = RANSAC(matches,100, features1, features2)
 
        #print(ssdRatio)
        final_image=image_first.copy()
        final_imag = cv2.drawMatches(image_first,features1,image_second,features2,matches,None)
        final_imag2 = cv2.drawMatches(image_first,features1,image_second,features2,matches_refined,None)
#        cv2.imshow("Matches",final_imag)
        
        cv2.imwrite("1b.png",cv2.drawKeypoints(img1, features1[:300],None,color=(0,255,0),flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS))
        cv2.imwrite("1c.png",cv2.drawKeypoints(img2, features2[:300],None,color=(0,255,0),flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS))
        cv2.imwrite("2.png",final_imag)
        cv2.imwrite("3.png",final_imag2)
        cv2.imwrite("4.png",finalimage)
        
    name = "Rainier" + str(first) + str(second)+ ".png"
    cv2.imwrite(name,finalimage)
    cv2.waitKey(0)

def User_images(first,second):
    
    name = "userclicked" + str(first)+".png"
    img1 = cv2.imread(name,cv2.COLOR_BGR2GRAY)  
    name = "userclicked" + str(second)+".png"      
    img2 = cv2.imread(name,cv2.COLOR_BGR2GRAY)  

    cv2.resize(img1,(512,512))
    cv2.resize(img2,(512,512))
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
        
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches_refined, hom, hom_inv = RANSAC(matches,100, kp1, kp2)
            
    finalimage = stitch(img1, img2, hom, hom_inv)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None)
    img4 = cv2.drawMatches(img1,kp1,img2,kp2,matches_refined,None)
    
    name = "userclicked" + str(first) + str(second)+ ".png"
    cv2.imwrite(name,finalimage)
    print("Done user Images")
    cv2.waitKey(0)

def Main_project():
    cb, ab, mb = detectCorner("Boxes.png")
    fb, nonmaxb = nonmaxSuppression(cb, ab ,cv2.imread("Boxes.png"))
    cv2.imwrite("1a.png",cv2.drawKeypoints(cv2.imread("Boxes.png"), fb,None,color=(0,255,0),flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS))
    print("Boxes.png corner detected")
    main(1,2) 
    print("Stitched image 1 and 2")
    main(12,3)
    print("Stitched Previous and 3")
    main(123,4)
    print("Stitched Previous and 4")
    main(1234,5)
    print("Stitched Previous and 5")
    main(12345,6)
    print("Stitched Previous and 6")
    print("Stitched all images")

         
def User_project():
    
    User_images(1,2)
    User_images(12,3)

Main_project() 
User_project()
threshold=35000    