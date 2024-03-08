import cv2
import numpy as np

FOCAL = 910
RES = (1164, 874)

def process_and_flow(prev, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, frame, flow=None, pyr_scale=0.25, levels=1, winsize=10,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=None)
    return frame, cv2.cvtColor(flow, cv2.COLOR_RGB2GRAY)

def get_edges(prev, frame):
    horiz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)

    med = np.median(horiz)
    sigma = int(np.std(horiz))
    upper = int(min(255, med + 2*sigma))
    lower = int(max(0, med - 2*sigma))

    _, thresh = cv2.threshold(horiz, 127, 255, cv2.THRESH_BINARY)
    thresh = 255 * thresh.astype(np.uint8)
    thresh = cv2.Canny(thresh, lower, upper)
    return thresh

def crop(edge):
    L_edge = edge[0:RES[1], 0:int(RES[0]/2)]
    R_edge = edge[0:RES[1], int(RES[0]/2):]
    return L_edge, R_edge

if __name__ == "__main__":
    pitch = [0]
    yaw = [0]
    cap = cv2.VideoCapture('labeled/0.hevc')
    prev = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            #Pre-processing
            frame, flow = process_and_flow(prev, frame)

            #Edge and Line Detection
            #edges = get_edges(prev, frame)
            prev = frame.copy()

            #Identify Edges
            #L_edge, R_edge = crop(edges)

            #L = cv2.HoughLinesP(L_edge, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            #R = cv2.HoughLinesP(R_edge, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

            #Calculate pitch and yaw


            #Show video for debugging
            cv2.imshow('flow', flow)
            cv2.waitKey(1)

        else:
            break
    
    cv2.destroyAllWindows()
    cap.release()