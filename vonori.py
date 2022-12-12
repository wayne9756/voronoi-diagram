# $LAN=PYTHON$
from cmath import acos
import math
import copy
from matplotlib.pyplot import draw
import numpy as np
import tkinter as tk
import tkinter.messagebox as msg
from tkinter.filedialog import askopenfilename
from operator import itemgetter, attrgetter
from sympy import true
#建立視窗
win = tk.Tk()
win.title("Vonori Diagram")
win.geometry('1200x960')
win.resizable(False,False)

class Equation:
    eq = None
    c = None
    m = None
    b = None
    def __init__(self,m,b):
        self.m = m
        self.b = b
        self.eq = np.array([m,-1])
        self.c = np.array([b])
    def intersection(self,x = None,y = None):
        if(x != None):
            return self.m*x - self.b
        else:
            return (self.b+y)/self.m

class pEquation:
    x0 = None
    y0 = None
    a = None
    b = None
    def __init__(self,a,b,x0,y0):
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b

class Hyperplane:
    p1 = None
    p2 = None
    p3 = None
    p4 = None
    eq = None
    n1 = None
    n2 = None
    def __init__(self,p1,p2,p3,p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.mid_x = (p1.x+p2.x)/2
        self.mid_y = (p1.y+p2.y)/2
        if(self.p1.y != self.p2.y):
            slope = -1/get_slope(p1, p2)
            self.eq = Equation(slope,slope*self.mid_x-self.mid_y) #equation list
            self.n1 = Vertex(x = self.eq.intersection(y = -1), y = -1)
            self.n2 = Vertex(x = self.eq.intersection(y = 1000), y = 1000)
        else:
            self.n1 = Vertex(x = self.mid_x, y = -1)
            self.n2 = Vertex(x = self.mid_x, y = 1000)
        # diagram.create_line(mid_x,mid_y,eq.intersection(y = -1),-1,fill = 'red',width = 5)

    def step_one(self,left,right):
        left_edge = left.edgeList[0]
        right_edge = right.edgeList[0]
        left_section = intersect_point(self.n1,self.n2,left_edge.p1,left_edge.p2) 
        right_section = intersect_point(self.n1,self.n2,right_edge.p1,right_edge.p2)
        if(left_section.y > right_section.y): # 碰到右半邊要消左半邊
            diagram.create_line(right_edge.p1.x,right_edge.p1.y,right_section.x,right_section.y,width = 8, fill = 'black')
            diagram.create_line(self.n1.x,self.n1.y,right_section.x,right_section.y,width = 5, fill='blue',tags = 'seg1')
            return 0, Vertex(2*right_edge.mid.x - self.p2.x,2*right_edge.mid.y - self.p2.y)
        else: #碰到左半邊，要消右半邊
            print("碰到左半邊，要消右半邊")
            diagram.create_line(left_edge.p2.x,left_edge.p2.y,left_section.x,left_section.y,width = 6, fill = 'black')
            diagram.create_line(self.n1.x,self.n1.y,left_section.x,left_section.y,width = 5, fill = 'blue',tags = 'seg1')
            return 1,Vertex(2*left_edge.mid.x - self.p1.x,2*left_edge.mid.y - self.p1.y)

    def step_two(self,left , right ,flag):
        left_edge = left.edgeList[0]
        right_edge = right.edgeList[0]
        left_section = intersect_point(self.n1,self.n2,left_edge.p1,left_edge.p2) 
        right_section = intersect_point(self.n1,self.n2,right_edge.p1,right_edge.p2)
        if flag  == 0: # 消右邊
            x = diagram.create_line(left_section.x,left_section.y,left_edge.p2.x,left_edge.p2.y,fill = 'black', width = 6)
            diagram.create_line(left_section.x,left_section.y,right_section.x,right_section.y,width= 6, fill = 'blue')
            return Vertex(left_section.x,left_section.y)
        else: # 消左邊
            diagram.create_line(right_section.x,right_section.y,right_edge.p1.x,right_edge.p1.y,fill = 'black', width = 6)
            diagram.create_line(left_section.x,left_section.y,right_section.x,right_section.y,width= 6, fill = 'blue')
            return Vertex(right_section.x,right_section.y)

    def step_three(self,p):
        diagram.create_line(p.x,p.y,self.eq.intersection(y=1000),1000,fill = 'green',width = 6)

            
class VD:
    pointList = None
    edgeList = None
    def __init__(self,pointList, edgeList):
        self.pointList = pointList
        self.edgeList = edgeList
    def __add__(self,item):
        p  = self.pointList + item.pointList
        e = self.edgeList + item.edgeList
        return VD(pointList=p,edgeList=e)
        
class Edge:
    p1 = None
    p2 = None
    mid = None
    def __init__(self,x1,y1,x2,y2,mid):
        self.p1 = Vertex(x1,y1)
        self.p2 = Vertex(x2,y2)
        self.mid = mid

class Vertex:
    x = None
    y = None
    v = None
    angle = None
    def __init__(self,x,y,v = None,angle = None):
        self.x = x
        self.y = y
        self.v = v

global diagram,pointlist,flag,num_loop,point_temp, step_count,fflag
fflag = 0
num_loop = []
point_temp = []
pointlist = []
edgelist = []#實際上還是存點
step_count = 0
def intersection(eq1, eq2, x = False, y = False):
    A = np.array([
        [eq1.a, - eq2.a],
        [eq1.b, - eq2.b]
    ])
    B = np.array([
        [eq2.x0 - eq1.x0],
        [eq2.y0 - eq1.y0]

    ])
    A_inv = np.linalg.inv(A)
    ans = A_inv.dot(B) #ans[0] is eq1 的 t 
    if x == True:
        return eq1.x0+eq1.a*float(ans[0]) # (x, y) insection
    else:
        return eq1.y0 + eq1.b * float(ans[0]) # (x, y) insection
def v_cross(v1, v2):
    return v1.x * v2.y - v1.y * v2.x

def intersect_point(a1,a2,b1,b2):
    a = Vertex(x = (a2.x -a1.x), y = (a2.y - a1.y))
    b = Vertex(x = (b2.x - b1.x), y = (b2.y - b1.y))
    s = Vertex(x = (b1.x - a1.x), y = (b1.y - a1.y))
    p = Vertex(a1.x+a.x*v_cross(s,b)/v_cross(a,b),a1.y+a.y*v_cross(s,b))
    return Vertex(a1.x+a.x*v_cross(s,b)/v_cross(a,b),a1.y+a.y*v_cross(s,b)/v_cross(a,b))


def show():
    count = 0
    for i in range (len(data) -1):
        if 'P' in data[i]:
            count += 1
            x , y = data[i][2:].split(' ')[0],data[i][2:].split(' ')[1]
            print("x = ", x , "y = ", y)
            point_temp.append(Vertex(int(x),int(y)))
    num_loop.append(count)
    if count == 2:
        two_point(0)
    else:
        three_point(0)

def readfile():
    global data
    tf = askopenfilename(title = "Plase choose a file",filetypes = (("Text Files","*.txt"),))
    tf = open(tf,'r')
    data = tf.read().splitlines()
    # Remove commments
    if 'P' in data[0]:
        show()
    else:
        for i in range(len(data)-1, -1,-1):
            if '#' in data[i]:
                data.pop(i)
            if data[i] == '':
                data.pop(i)
        print(data)
        for word in data:
            if ' ' not in word:
                data.remove(word)
                num_loop.append(int(word))
        for word in data:
            x , y = word.split(' ')[0],word.split(' ')[1]
            pointlist.append(Vertex(int(x),int(y)))
        print(num_loop)
        next_set.counter = 0

def normalize(vect):
    return vect/np.linalg.norm(vect)


def create_but(name,x,command=None):
    but = tk.Button(win,text=name ,command = command,bg = 'gray', fg = 'black', font= ("Arial",12))
    but['width'] = 15 
    but['height'] = 4
    but['activebackground'] = 'red'
    but['activeforeground'] = 'yellow'
    but.grid(row = 0,column = x)
#create button   

def intersect1d(a1, a2, b1, b2) -> bool :
    if(a1 > a2 ):
        a1, a2 = a2, a1
    if(b1 > b2):
        b1, b2 = b2, b1
    return max(a1,b1) <= min(a2,b2)

def sign(value):
    return 0 if(abs(value) < math.exp(-1)) else 1 if(value > 0) else -1

def intersect(a1, a2, b1, b2) -> bool:
    return intersect1d(a1.x, a2.x, b1.x, b2.x) and intersect1d(a1.y, a2.y, b1.y, b2.y) and (sign(cross(a1, a2, b1)) * sign(cross(a1, a2, b2)) <= 0) and (sign(cross(b1,b2,a1)) * sign(cross(b1,b2,a2)) <= 0)



def is_line(List):
    area = get_area(List)
    v1, v2, v3 = tuple(List)
    if area == 0: # 三點共線
        print("is line")
        if(v1.y == v2.y and v2.y == v3.y):#平行x軸
            List.sort(key = lambda s : s.x)
            line1_x = (List[0].x + List[1].x)/2
            line2_x = (List[1].x + List[2].x)/2
            edgelist.append(Vertex(x = line1_x, y = -1))
            edgelist.append(Vertex(x = line1_x, y = 1000))
            edgelist.append(Vertex(x = line2_x, y = -1))
            edgelist.append(Vertex(x = line2_x, y = 1000))
            diagram.create_line(line1_x,-1,line1_x,1000,fill = 'red',width = 5)
            diagram.create_line(line2_x,-1,line2_x,1000,fill = 'red',width = 5)
            return True
        elif(List[0].x == List[1].x and List[1].x == List[2].x):#平行y軸
            List.sort(key = lambda s : s.y)
            line1_y = (List[0].y + List[1].y)/2
            line2_y = (List[1].y + List[2].y)/2
            edgelist.append(Vertex(x = -1, y = line1_y))
            edgelist.append(Vertex(x = 1000, y = line1_y))
            edgelist.append(Vertex(x = -1, y = line2_y))
            edgelist.append(Vertex(x = 1000, y = line2_y))
            diagram.create_line(-1,line1_y, 1000,line1_y,fill = 'red', width = 5)
            diagram.create_line(-1,line2_y,1000,line2_y,fill = 'red', width = 5)
            return True
        else:
            List.sort(key = lambda s : s.x)
            for i in range(0,2):
                mid_x = (List[i+1].x + List[i].x)/2
                mid_y = (List[i+1].y + List[i].y)/2
                slope = -1/get_slope(List[i+1],List[i])
                eq = Equation(slope,slope*mid_x-mid_y)
                if(abs(slope) > 10):
                    edgelist.append(Vertex(x = eq.intersection(y = 1000),y = 1000))
                    edgelist.append(Vertex(x = eq.intersection(y = -1), y = -1))
                    diagram.create_line(eq.intersection(y = 1000), 1000,eq.intersection(y = -1),-1, fill='red', width = 5)
                else:
                    edgelist.append(Vertex(x = -1, y = eq.intersection(x = -1)))
                    edgelist.append(Vertex(x = 1000,y = eq.intersection(x = 1000)))
                    diagram.create_line(-1,eq.intersection(x = -1),1000,eq.intersection(x = 1000), fill = 'red', width = 5)
            return True
    else :
        print("not line")
        return False

def two_point(List):
    edgelist.clear()
    v1, v2 = tuple(List)
    mid_x = (v1.x+v2.x)/2
    mid_y = (v1.y+v2.y)/2
    if(v1.x == v2.x and v1.y == v2.y) :
        print('illegel case')
    elif( v2.x != v1.x and v2.y != v1.y): # 不共線
        slope = -1/get_slope(v1, v2)
        eq = (Equation(slope,slope*mid_x-mid_y))#equation list
        if(abs(slope) > 6):
            edgelist.append(Edge(x2 = eq.intersection(y = -1), y2 = -1, x1 = eq.intersection(y = 1000), y1 = 1000, mid = Vertex(mid_x, mid_y)))
        else:
            edgelist.append(Edge(y1 = eq.intersection(x = -1), x1 = -1, y2 = eq.intersection(x = 1000), x2 = 1000, mid = Vertex(mid_x, mid_y)))
    elif(v1.x == v2.x): # 兩點 x 平行
            edgelist.append(Edge(x1 = -1, y1 = mid_y, x2 = 1000, y2 = mid_y, mid = Vertex(mid_x, mid_y)))
    elif(v1.y == v2.y):
            edgelist.append(Edge(x1 = mid_x, y1 = -1, x2 = mid_x, y2 = 1000, mid = Vertex(mid_x, mid_y)))
    diagram.create_line(v1.x,v1.y,v2.x,v2.y,dash = (4,4),fill = 'green',width = 3)
    diagram.create_line(edgelist[0].p1.x ,edgelist[0].p1.y,edgelist[0].p2.x,edgelist[0].p2.y ,fill='red', width = 5)
    return VD(edgeList = copy.copy(edgelist), pointList = copy.copy(List))

def three_point(List):
    v1, v2, v3 = tuple(List)
    diagram.create_oval(v1.x-5, v1.y-5, v1.x+5, v1.y+5, fill='white')
    diagram.create_oval(v2.x-5, v2.y-5, v2.x+5, v2.y+5, fill='white')
    diagram.create_oval(v3.x-5, v3.y-5, v3.x+5, v3.y+5, fill='white')
    get_center(List)
    return draw_line(List)

def draw_point(event):
    point = Vertex(event.x,event.y)
    pointlist.append(point)
    diagram.create_oval(point.x-5, point.y-5, point.x+5, point.y+5, fill='white')

def next_set():
    global pointlist ,edgelist
    diagram.delete('all')
    edgelist.clear()
    index = next_set.counter
    print("index = ", index)
    start = sum(num_loop[0:index])#the start position in pointlist array
    if num_loop[index] == 2:
        List = pointlist[start:start+2]
        two_point(List)
    elif num_loop[index] == 3:
        List = pointlist[start:start+3]
        three_point(List)
    else :
        List = pointlist[start:start+4]
        diagram.create_line(List[0].x,-5,List[0].y-5,List[0].x+5,List[0].y+5,fill = 'white')
        diagram.create_line(List[1].x,-5,List[1].y-5,List[1].x+5,List[1].y+5,fill = 'white')
        diagram.create_line(List[2].x,-5,List[2].y-5,List[2].x+5,List[2].y+5,fill = 'white')
        diagram.create_line(List[3].x,-5,List[3].y-5,List[3].x+5,List[3].y+5,fill = 'white')
        voronoi(List)
    next_set.counter += 1

def get_area(List):
    v1, v2, v3 = tuple(List)
    A = np.array([
        [v1.x, v1.y, 1],
        [v2.x, v2.y, 1],
        [v3.x, v3.y, 1]
    ])
    return np.linalg.det(A)*0.5


def clean_canvas():
    global step_count, fflag
    fflag = 0
    step_count = 0 
    diagram.delete("all")
    pointlist.clear()
    point_temp.clear()
    num_loop.clear()
    edgelist.clear()
# Main function to create vonori diagram
def get_slope(a,b):
    return ((a.y-b.y)/(a.x-b.x+math.exp(0.00000000001)))

def get_center(List):
    area = get_area(List)
    X = np.array([
        [(List[0].x**2 + List[0].y**2),List[0].y,1],
        [(List[1].x**2 + List[1].y**2),List[1].y,1],
        [(List[2].x**2 + List[2].y**2),List[2].y,1]
    ])
    Y = np.array([
        [List[0].x, (List[0].x**2 + List[0].y**2),1],
        [List[1].x, (List[1].x**2 + List[1].y**2),1],
        [List[2].x, (List[2].x**2 + List[2].y**2),1]
    ])
    O = (float(np.linalg.det(X)/(4*area)), float(np.linalg.det(Y)/(4*area)))
    # diagram.create_oval(O[0]-5,O[1]-5,O[0]+5,O[1]+5, fill = 'red')
    return float(np.linalg.det(X)/(4*area)), float(np.linalg.det(Y)/(4*area))

def draw_line(List):
    eqlist = []
    edgelist.clear()
    if (not is_line(List)):
        G = Vertex((List[0].x+List[1].x+List[2].x)/3 , (List[0].y+List[1].y+List[2].y)/3)#重心
        for i in range(0,3):
            if(List[i].y - G.y) > 0 :
                List[i].angle = math.acos((List[i].x-G.x)/np.linalg.norm(np.array([List[i].x-G.x , List[i].y-G.y]))) # 依照angle 對點逆時針排序
            else:
                List[i].angle = 2*math.pi - math.acos((List[i].x-G.x)/np.linalg.norm(np.array([List[i].x-G.x , List[i].y-G.y])))
        List.sort(key = lambda s : s.angle)
        (center_x, center_y) = get_center(List)
        # diagram.create_oval(center_x+5,center_y+5,center_x-5,center_y-5,fill = 'red')
        x_buttom = pEquation(a = 0,b = 1,x0 = -1,y0 =-1000)
        x_top = pEquation(a = 0, b = 1, x0 = 1000, y0 =-1000)
        y_buttom = pEquation(a = 1, b = 0,x0 = -1000, y0 = -1)
        y_top = pEquation(a = 1, b = 0, x0 = -1000, y0 = 1000)
        for i in range(0,3):
            mid_x = (List[i].x+List[(i+1)%3].x)/2
            mid_y = (List[i].y+List[(i+1)%3].y)/2
            n = (List[(i+1)%3].y-List[i].y, List[i].x-List[((i+1)%3)].x)
            peq = pEquation(a = n[0], b = n[1],x0 = mid_x, y0 = mid_y)
            # eqlist.append()
            if(n[1] < 0 and abs(n[1]) > 6):
                edgelist.append(Vertex(center_x,center_y))
                edgelist.append(Vertex(x = intersection(peq,y_buttom,x = True),y = intersection(peq,y_buttom,y=True)))
                diagram.create_line(center_x,center_y,intersection(peq,y_buttom,x = True),intersection(peq,y_buttom,y=True), fill='red', width = 5)
                print("case 1")
            elif(n[1] < 0 and n[0] > 0):
                edgelist.append(Vertex(center_x,center_y))
                edgelist.append(Vertex(x = intersection(peq,x_top,x=True), y = intersection(peq,x_top,y=True)))
                diagram.create_line(center_x,center_y,intersection(peq,x_top,x = True),intersection(peq,x_top, y = True), fill = 'red', width = 5)
                print("case 2")
            elif(n[1] < 0 and n[0] < 0):
                edgelist.append(Vertex(center_x,center_y))
                edgelist.append(Vertex(x = intersection(peq,x_buttom,x=True), y = intersection(peq,x_buttom,y=True)))
                diagram.create_line(center_x,center_y,intersection(peq,x_buttom,x = True),intersection(peq,x_buttom,y=True), fill = 'red', width = 5)
                print("case 3")
            elif(n[1] >= 0 and abs(n[1]) > 6):
                edgelist.append(Vertex(center_x,center_y))
                edgelist.append(Vertex(x = intersection(peq,y_top,x=True), y = intersection(peq,y_top,y=True)))
                diagram.create_line(center_x,center_y,intersection(peq,y_top,x = True),intersection(peq,y_top,y = True), fill = 'red', width = 5)
                print("case 4")
            elif(n[1] >= 0 and n[0] > 0):
                edgelist.append(Vertex(center_x,center_y))
                edgelist.append(Vertex(x = intersection(peq,x_top,x=True), y = intersection(peq,x_top,y=True)))
                diagram.create_line(center_x,center_y,intersection(peq, x_top,x = True), intersection(peq,x_top, y = True),fill = 'red', width = 5)
                print("case 5")
            elif(n[1] >= 0 and n[0] < 0):
                edgelist.append(Vertex(center_x,center_y))
                edgelist.append(Vertex(x = intersection(peq,x_buttom,x=True), y = intersection(peq,x_buttom,y=True)))
                diagram.create_line(center_x,center_y,intersection(peq, x_buttom,x = True),intersection(peq, x_buttom, y = True), fill = 'red', width = 5)
                print("case 6")
            else:
                print("case missing")
        return VD(edgeList = copy.deepcopy(edgelist), pointList  = copy.deepcopy(List))
    
def output():
    file = open('output.txt', 'w')
    pointlist.sort(key = lambda s: (s.x, s.y))
    if len(pointlist) == 2:
        edgelist.sort(key = lambda s:(s.x, s.y))
        for p in pointlist:
            print('P ' + str(p.x) +' '+ str(p.y),file=file)
        print('E ', str(edgelist[0].x) +' ' + str(edgelist[0].y) + ' ' + str(edgelist[1].x) + ' '+ str(edgelist[1].y) , file = file)
    else:
        for p in pointlist:
            print('P ' + str(p.x) +' '+ str(p.y),file=file)
        if not is_line():
            for i in range(0,3):
                n = sorted(edgelist[2*i:2*(i+1)],key = attrgetter('x','y'))
                print('E',str(n[0].x) + ' ' + str(n[0].y) + ' ' + str(n[1].x) + ' ' +  str(n[1].y),file = file)
        else:
            for i in range(0,2):
                n = sorted(edgelist[2*i:2*(i+1)],key = attrgetter('x','y'))
                print('E',str(n[0].x) + ' ' + str(n[0].y) + ' ' + str(n[1].x) + ' ' +  str(n[1].y),file = file)

def cross( o, a, b):
    return (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x)

def divide(pointList): # 會進入到divide length 一定是大於3
    length = len(pointList)
    temp = copy.deepcopy(pointList)
    mid = length//2
    return temp[:mid], temp[mid:]

def merge(left, right):
    global hyperplane,flag,p
    upper, lower = convexhull(left = left,right = right) # find the convexhull
    # 找出上切線及下切線
    for i in range(len(upper)):
        if upper[i] in left.pointList and upper[i+1] in right.pointList:
            up_left = upper[i]
            up_right = upper[i+1]
            break
            
    for i in range(len(lower)):
        if lower[i] in left.pointList and lower[i+1] in right.pointList:
            lower_left = lower[i]
            lower_right = lower[i+1]
            break
    if fflag :
        diagram.create_line(lower_left.x,lower_left.y,lower_right.x,lower_right.y,dash = (4,4),width = 3,fill = 'white',tags = 'first')
        print('execute')
        hyperplane = Hyperplane(lower_left, lower_right, up_left,up_right)
        flag , p = hyperplane.step_one(left,right)
        if flag :
            hyperplane = Hyperplane(p, lower_right,up_left, up_right)
        else:
            hyperplane = Hyperplane(lower_left, p, up_left, up_right)
        p = hyperplane.step_two(left,right,flag = flag)
    # if len([x for x i:n upper if x is not None]) == len([y for y in lower if y is not None]):
        # print('execute')
        diagram.create_line(up_left.x,up_left.y,up_right.x,up_right.y,dash = (4,4),width = 3, fill = 'white')
        hyperplane = Hyperplane(up_left,up_right,up_left,up_right)
        hyperplane.step_three(p = p)
    else:
        if step_count == 2:
            diagram.create_line(lower_left.x,lower_left.y,lower_right.x,lower_right.y,dash = (4,4),width = 3,fill = 'white',tags = 'first')
        elif step_count == 3 :
            print('execute')
            hyperplane = Hyperplane(lower_left, lower_right, up_left,up_right)
            flag , p = hyperplane.step_one(left,right)
            if flag :
                hyperplane = Hyperplane(p, lower_right,up_left, up_right)
            else:
                hyperplane = Hyperplane(lower_left, p, up_left, up_right)
        elif step_count == 4:
            p = hyperplane.step_two(left,right,flag = flag)
        # if len([x for x i:n upper if x is not None]) == len([y for y in lower if y is not None]):
            # print('execute')
        elif step_count == 5:
            diagram.create_line(up_left.x,up_left.y,up_right.x,up_right.y,dash = (4,4),width = 3, fill = 'white')
            hyperplane = Hyperplane(up_left,up_right,up_left,up_right)
            hyperplane.step_three(p = p)

    # up_mid = Vertex((up_left.x + up_right.x)/2, (up_left.y + up_right.y)/2)
    # lower_mid = Vertex((lower_left.x + lower_right.x)/2, (lower_left.y + lower_right.y)/2)
    # diagram.create_oval(up_mid.x-5, up_mid.y-5, up_mid.x+5, up_mid.y+5, fill='white')
    # diagram.create_oval(lower_mid.x-5, lower_mid.y-5, lower_mid.x+5, lower_mid.y+5, fill='white')


            
def voronoi(pointList):
    pointList.sort(key = attrgetter('x', 'y')) # 先對 x 排序在對 y 排序
    if(len(pointList) == 1 ):
        return
    elif(len(pointList) == 2):
        result = two_point(pointList)
    else:
        p_left , p_right = divide(pointList)
        v_left = voronoi(p_left) # 得到 左邊的voronoi diagram
        v_right = voronoi(p_right) # 得到 右邊的voronoi diagram
        result = merge(v_left, v_right) #合併整個 voronoi diagram  
    return result # return 整個voronoi diagram

def convexhull(left, right):
    tempd = left + right 
    tempd.pointList.sort(key = attrgetter('x', 'y'))
    temp = tempd.pointList
    length = len(temp)
    l = 0
    u = 0
    U = [None]*(length)
    L = [None]*(length)
    for i in range(0,(length)):
        while l >= 2 and (cross(L[l-2], L[l-1], temp[i]) <= 0):
            l-=1
        while u >= 2 and (cross(U[u-2], U[u-1], temp[i]) >= 0):
            u-=1
        print("i = ", i, " u = ", u, " l = ", l)
        U[u] = temp[i]
        L[l] = temp[i]
        l+=1
        u+=1
    for i in range(0,len(U)):
        if U[i] != None:
            point = U[i]
            print(i,' x = ', U[i].x, ' y = ', U[i].y)
            diagram.create_oval(point.x-5, point.y-5, point.x+5, point.y+5, fill='green')
    for i in range(0,len(L)):
        if L[i] != None:
            point = L[i]
            print(i,' x = ', L[i].x, ' y = ', L[i].y)
            diagram.create_oval(point.x-5, point.y-5, point.x+5, point.y+5, fill='red')
    return U, L
def next_step():
    global step_count,p_right,v_left,v_right,p_left
    pointList = pointlist
    pointList.sort(key = attrgetter('x', 'y')) # 先對 x 排序在對 y 排序
    if step_count == 0:
        p_left , p_right = divide(pointList)
        v_left = voronoi(p_left) # 得到 左邊的voronoi diagram
    elif step_count == 1:
        v_right = voronoi(p_right) # 得到 右邊的voronoi diagram
    else:
        result = merge(v_left, v_right) #合併整個 voronoi diagram  
    step_count += 1
def run():
    global fflag
    fflag = 1   
    voronoi(pointList = pointlist)

diagram = tk.Canvas(win, width=800,height = 800,bg = 'black')
diagram.grid(row=1,column=0,columnspan=5,sticky='we')
diagram.bind("<Button-1>",draw_point)
#define button
bt_read = create_but("Read Data",0,readfile)
bt_run = create_but("Run",1,run)
bt_step = create_but("Save",2,output)
bt_next = create_but("Next Set",3,next_set)
bt_nextt = create_but("step by step",5,next_step)
bt_clean = create_but("Clean", 4,clean_canvas)

win.mainloop()
