"""
Created on Mon Mar 25 20:07:44 2024

@author: ianva
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

CWD = os.path.dirname(os.getcwd())
SAVE_PATH = os.path.join(CWD, 'DBimages')
SAVE_PATH2 = os.path.join(CWD, 'LOOK_TAG')
TRU_DIR = os.path.join(CWD, r'DB_sections')
MERGE_DIR = os.path.join(CWD,r'MERGE')
TRU_PATH = os.path.join(CWD, r'DB_sections\SPY_TRU.csv')
DATA = os.path.join(CWD,"RAW_DATA")
df = pd.read_csv(os.path.join(DATA,"ALL_DATA.csv"), skiprows=[0], header = None)
lent = df.shape[0]
df = df.iloc[0:lent]
df_plot = df.iloc[0:lent,[0,2]]
region_size = 15

def merge_data (path = MERGE_DIR):
    all_data = pd.read_csv(os.path.join(MERGE_DIR,os.listdir(path)[0]))
    
    
    for file in os.listdir(path)[1:]:
        file_path = os.path.join(MERGE_DIR,file)
        temp = pd.read_csv(file_path)
        all_data = pd.concat([all_data,temp],ignore_index=1)
        
    all_data.to_csv(os.path.join(DATA,"ALL_DATA.csv"))





def click_event(event, x, y, flags, params): 
    global Xcoord
    global Ycoord
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
       Xcoord = x
       Ycoord = y


def annotate():
    label = os.path.join(DATA,"ALL_DATA.csv")
    
    #if the file had data, start where you left off, if its empty start at 0
    try:
        df_startval = pd.read_csv(TRU_PATH)
        a = df_startval.iloc[-1,1]
        print("START:  ")
        print(a)
    except pd.errors.EmptyDataError:
        a = 0
    
   
   

    end = 0
    while end < lent: # for all data in the .csv
        peakType = 0;
        dataPointA = -1;
        dataPointB = -1;
    
        # Define start and end points of the display
        b = a+(region_size)
        end += region_size
        
        df_sect = np.array(df_plot.iloc[a:b]) #Copy the needed section from the full dataframe
        
        plt.figure(figsize=(15,6))# make the figure wider to more easily see data
        
        # Get rid of padding on x axis (this makes it easier to go from pixels to data values)
        ax = plt.gca();
        ax.set_xlim(0.0, (region_size)-1);
        #Create the plot and save is as a unique image
        
        plt.plot(range(0,region_size),df_sect[:,1]) 
        print("HERE")
        num = a//region_size
        temp_path = os.path.join(SAVE_PATH,str(num))
        plt.savefig(temp_path) # Save the matplot plot as a png
        
        # set up some info about the image size
        img = cv2.imread(os.path.join(temp_path+".png"))#Read in the png of the matplot graph
        imgHeight = img.shape[0]
        graphLength = (img.shape[1]-(img.shape[1]-972) - 135) #the length, in pixels, of the x axis
        imgLength = graphLength 

        print("HERE")

        # Show lines on image for region that will be recorded
        #cv2.line(img,(imgLength+137,0),(imgLength+137,imgHeight),(255,0,0),1)
        #cv2.line(img,((imgLength*2)+138,0),((imgLength*2)+138,imgHeight),(255,0,0),1)

        cv2.putText(img, "DATA STARTS AT: "+ str(a) + " DAYS FROM JAN 31 1993",fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=.5,
                    org=(10,30),color = (0,0,0),thickness=1)
        cv2.imshow("REGION", img) #Display the image
        
        # Code for the UI
        k = 0
        peakType = 0
        while(k != ord('q')): # While the image is shown and q is not pressed
            cv2.setMouseCallback('REGION', click_event) #Record the pixel coords of a click
            if k == ord('w'): # If w is pressed
                
                #Record the click coords as the start point and draw a line there
                XcoordA = Xcoord
                YcoordA = Ycoord
                cv2.line(img,(XcoordA,YcoordA - 10),(XcoordA,YcoordA+10),(0,255,0),2)
                cv2.imshow("REGION", img)
                peakType = 1
                dataPointA = (((XcoordA-135)/graphLength)*(region_size))//1
            elif k == ord('e'): # same for end point
                XcoordB = Xcoord
                YcoordB = Ycoord
                cv2.line(img,(XcoordB,YcoordB - 10),(XcoordB,YcoordB+10),(0,0,255),2)
                cv2.imshow("REGION", img)
                peakType = 1
                dataPointB = (((XcoordB-135)/graphLength)*(region_size))//1
                
            k =  cv2.waitKey(0) # always set K equal to whatever key is pressed
        
        # Convert pixel coords on the image to a datapoint in the csv
        
        
        
        #Close and clear the window
        cv2.destroyAllWindows() 
        plt.clf()
        os.remove(os.path.join(temp_path+".png")) #no need to keep images, delete them
        
        if a ==0:
            df_out = pd.DataFrame({"Start":a,
                    "End":[b],
                    "DBstart":[dataPointA],
                    "DBend":[dataPointB],
                    "Class":[peakType]})
            if peakType == 1:
                df_out.to_csv(os.path.join(TRU_DIR, "SPY_TRU_onlyDB.csv"), index=False)
            elif peakType == 0:
                df_out.to_csv(os.path.join(TRU_DIR, "SPY_TRU_neg.csv"), index=False)
        else:
            data_temp = pd.DataFrame({"Start":[a],
                    "End":[b],
                    "DBstart":[dataPointA],
                    "DBend":[dataPointB],
                    "Class":[peakType]})
            df_out = pd.read_csv(TRU_PATH)
            df_out = pd.concat([df_out,data_temp], ignore_index=1)
            
            if peakType == 1:
                try:
                    df_out_tru = pd.read_csv(os.path.join(TRU_DIR, 'SPY_TRU_onlyDB.csv'))
                    df_out_tru = pd.concat([df_out_tru,data_temp], ignore_index=1)
                    df_out_tru.to_csv(os.path.join(TRU_DIR, 'SPY_TRU_onlyDB.csv'), index=False)
                except pd.errors.EmptyDataError:
                    df_out.to_csv(os.path.join(TRU_DIR, 'SPY_TRU_onlyDB.csv'), index=False)
                
            elif peakType == 0:
                try:
                    df_out_neg = pd.read_csv(os.path.join(TRU_DIR, "SPY_TRU_onlyDB.csv"))
                    df_out_neg = pd.concat([df_out_neg,data_temp], ignore_index=1)
                    df_out_neg.to_csv(os.path.join(TRU_DIR, "SPY_TRU_neg.csv"), index=False)
                except pd.errors.EmptyDataError:
                    df_out.to_csv(os.path.join(TRU_DIR, 'SPY_TRU_onlyDB.csv'), index=False)
    

        df_out.to_csv(TRU_PATH, index=False)
        
        '''
        If the section had a double bottom, shift the whole section out. Now that section
        is the first third of the next image displayed.
        
        If it did not have a DB, only shift the earlier half out. This way if there was a
        DB on the edge of the boundry it will still be recorded 
        '''
        
        if peakType == 1:
            a += region_size
        else:
            a += region_size//2
            
          
def visualize_edit (tru_path = r"C:\Users\ianva\TechnicalAnalysisCNN\DB_sections\SPY_TRU.csv"):
    df_tru = pd.read_csv(tru_path)
   
    length = df_tru.shape[0]
    try:
        df_startval = pd.read_csv(r"C:\Users\ianva\TechnicalAnalysisCNN\DB_sections\SPY_TRU_UPDATED.csv")
        ref_start = df_startval.iloc[-1,0]
        print("START:  ")
        print(ref_start)
        count = ref_start
        count_index = ref_start
        end = ref_start
    except pd.errors.EmptyDataError:
        count = 0
        count_index = 0
        end = 0
    
        
    while count < length:
        #print(df_tru.iloc[count])
        start  = df_tru.iloc[count,0]
        end = df_tru.iloc[count,1]
        #print(start)
        #print(end)
        df_sect = np.array(df_plot.iloc[start:end])
        bar_start = df_tru.iloc[count,2]
        bar_end = df_tru.iloc[count,3]
        
        
        plt.figure(figsize=(15,6))# make the figure wider to more easily see data
        
        # Get rid of padding on x axis (this makes it easier to go from pixels to data values)
        ax = plt.gca();
        ax.set_xlim(0, region_size-1);
        
        
        plt.plot(range(0,region_size),df_sect[:,1]) 
        num = end//region_size
        temp_path = os.path.join(SAVE_PATH2,str(num))
        plt.savefig(temp_path) # Save the matplot plot as a png
        
        img = cv2.imread(os.path.join(temp_path+".png"))#Read in the png of the matplot graph
        imgHeight = img.shape[0]
        graphLength = (img.shape[1]-(img.shape[1]-972) - 135) #the length, in pixels, of the x axis



        x1 = int(((bar_start/region_size)*graphLength)+135)
        x2 = int(((bar_end/region_size)*graphLength)+135)
        #print(bar_start)
        #print(x1)
        print("    ")
    #   print(bar_end)
       # print(x2)
       
        
        cv2.line(img,(x1,0),(x1,imgHeight),(0,255,0),3)
        cv2.line(img,(x2,0),(x2,imgHeight),(0,0,255),3)
        
        cv2.imshow("region", img)
        print("current location:")
        print(start)
        
        k = cv2.waitKey(0)
       
        
        
        if k == ord('d'):
            #print(df_tru[count])
            df_tru = df_tru.drop([count_index])
            print("REMOVED")
        else:
            count += 1
        count_index += 1
        print("DF:")
        print(df_tru)
        print("--------------------------")
        df_tru.to_csv(r"C:\Users\ianva\TechnicalAnalysisCNN\DB_sections\SPY_TRU_UPDATED.csv")
        
def visualize (tru_path = r"C:\Users\ianva\TechnicalAnalysisCNN\DB_sections\SPY_TRU_augment.csv"):
    df_tru = pd.read_csv(tru_path)
   
    length = df_tru.shape[0]

    count = 0
    end = 0
    
        
    while count < length:
        #print(df_tru.iloc[count])
        start  = df_tru.iloc[count,0]
        end = df_tru.iloc[count,1]
        #print(start)
        #print(end)
        df_sect = np.array(df_plot.iloc[start:end])
        bar_start = df_tru.iloc[count,2]
        bar_end = df_tru.iloc[count,3]
        
        
        plt.figure(figsize=(15,6))# make the figure wider to more easily see data
        
        # Get rid of padding on x axis (this makes it easier to go from pixels to data values)
        ax = plt.gca();
        ax.set_xlim(0, region_size-1);
        x = df_sect[:,1]
        x = (x-x.min())
        x = x/x.max()
        
        plt.plot(range(0,region_size),x) 
        num = end//region_size
        temp_path = os.path.join(SAVE_PATH2,str(num))
        plt.savefig(temp_path) # Save the matplot plot as a png
        
        img = cv2.imread(os.path.join(temp_path+".png"))#Read in the png of the matplot graph
        imgHeight = img.shape[0]
        graphLength = (img.shape[1]-(img.shape[1]-972) - 135) #the length, in pixels, of the x axis



        x1 = int(((bar_start/region_size)*graphLength)+135)
        x2 = int(((bar_end/region_size)*graphLength)+135)
        #print(bar_start)
        #print(x1)
        print("    ")
    #   print(bar_end)
       # print(x2)
       
        
        cv2.line(img,(x1,0),(x1,imgHeight),(0,255,0),3)
        cv2.line(img,(x2,0),(x2,imgHeight),(0,0,255),3)
        
        cv2.imshow("region", img)
        print("current location:")
        print(start)
        
        cv2.waitKey(0)
       
        count += 1
        
        
        
def augment_slide(tru_path = r"C:\Users\ianva\TechnicalAnalysisCNN\DB_sections\SPY_TRU_onlyDB.csv", data_path = os.path.join(DATA,"ALL_DATA.csv")):
    df_tru = pd.read_csv(tru_path)
    #df_data = pd.read_csv(data_path,usecols=[2],skiprows=[0])
    count = 0
    size = df_tru.shape[0]
    while count < size:
        
        start_orig, end_orig = df_tru.iloc[count,[0,1]]
        start = start_orig+2
        end = end_orig+2
        # = df_data.iloc[start:end]
        print("augmenting db at pos " + str(start) + "   "+ str(end))
        #print(data)
        db_indexA = df_tru.iloc[count,2]
        db_indexB = df_tru.iloc[count,3]
        #db_start = start+db_indexA
        db_end = start_orig+db_indexB
        print("DB end = " +str(db_end))
        while db_end < end-2:
            start -= 1
            end -=1
            if start != start_orig:
                df_temp = pd.DataFrame({"Start":[start],
                        "End":[end],
                        "DBstart":[db_indexA],
                        "DBend":[db_indexB],
                        "Class":[1]})
                #print(df_temp)
                df_tru = pd.concat([df_tru,df_temp])
                df_tru.to_csv(r"C:\Users\ianva\TechnicalAnalysisCNN\DB_sections\SPY_TRU_augment.csv",index = False)
        count +=1
    

        
#annotate()  
#visualize()
#merge_data()
#augment_slide()














