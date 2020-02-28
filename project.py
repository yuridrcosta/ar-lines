import urllib.request
import cv2
import numpy as np
import glob


cap = cv2.VideoCapture(0)
webcam_URL = "http://192.168.15.4:8080/shot.jpg"

chessboard_x = 4
chessboard_y = 4
# Criteria são critérios que o opencv usa para definir quantas iterações de calibração serão executadas
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_x*chessboard_y,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_x,0:chessboard_y].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)
#axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

objpoints = [] # Guarda os pontos 3d no mundo real
imgpoints = [] # Guarda os pontos em 2d no plano da imagem
points_counter = 0
ret = False


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # Desenhando o chão
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # Desenhando as linhas verticais
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # Desenhando o teto    
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img    
def draw_function(img,imgpts):
	for i in range(120):	
		img = cv2.line(img, tuple(imgpts[i].ravel()), tuple(imgpts[i+1].ravel()), (255), 5)
	return img    

def read_function(func,t):  #função que identifica as funções matemáticas da entrada
    if func == 'sen(t)' :
        return np.sin(t)
    if func == 'cos(t)' :
        return np.cos(t)
    if func == 'tan(t)' :
        return np.tan(t)
    if func == 'ln(t)':
        return np.log(t)
    else:
        lista = list(func)
        if len(lista) == 3:
            num1 = t
            num2 = float(lista[2])
            sinal = lista[1]
            
            if sinal == '+':
                return num1+num2
            if sinal == '-':
                return num1-num2
            if sinal == '*':
                return num1*num2
            if sinal == '/':
                return num1/num2
            if sinal == '^':
                return num1**num2
        else:
            return float(func)
    
print("Digite a fonte da imagem: [1 para Webcam 2 para smartphone]")
img_source = int(input("> "))
inp = input("Digite a curva em função da variável t sem o intervalo definido separando cada função por uma virgula (nomenclatura em português): Ex.: cos(t),sen(t),3\n>")
splited_input = inp.split(",") #separa a entrada em um array que divide cada palavra 
x = splited_input[0] # recebe a função que vai representar o eixo x na curva
y = splited_input[1] # recebe a função que vai representar o eixo y na curva
z = splited_input[2] # recebe a função que vai representar o eixo z na curva
times = 1 
while True:
	if img_source == 2:
		imgR=urllib.request.urlopen(webcam_URL)
		imgNp=np.array(bytearray(imgR.read()),dtype=np.uint8)
		img = cv2.imdecode(imgNp,-1)
	else:
		ret1,img = cap.read()
	cv2.imshow("Webcam",img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (chessboard_x,chessboard_y),None)
	if ret == True:

		objpoints.append(objp)
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(1,1),criteria)
		imgpoints.append(corners2)
		points_counter+=1
		if points_counter == 5:
			# mtx é a matriz da camera,dist são os coeficientes de distorção
			# rvecs é o vetor de rotação (usa a rodrigues function para converter em matriz)
			# tvecs é o vetor de translação 
			ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		elif points_counter>5:
			#print("corners2: {}\n".format(corners2))
			
			#corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			# Calcular a nova posição da camera a partir das novas imagens
			_,rvecs, tvecs,_ = cv2.solvePnPRansac(objp, corners2, mtx, dist,True,cv2.SOLVEPNP_EPNP)
			# Calcularemos os pontos que devem fazer parte da curva e em seguidas acharemos seus respectivos no plano da imagem

			# Aqui estamos definindo quais são esses pontos que queremos encontrar
			points3d = []
			for i in range(150):
				t = i/5
				#points3d.append([2,2,np.sin(t)])
				points3d.append([read_function(x,t),read_function(y,t),read_function(z,t)]) # Isso aqui que está dentro do parenteses é a curva.
			points = np.asarray(points3d).reshape(-1,3)
			# Essa função acha os pontos na imagem que são correspondentes aos pontos no mundo 3d
			points2d, jac = cv2.projectPoints(points, rvecs, tvecs, mtx, dist)
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
			

			# Obter matriz de rotação a partir de vetor de rotação
			#rmtx,_ = cv2.Rodrigues(rvecs)
			# Concatenar matriz de rotação e vetor de translação
			#Rt = np.concatenate((rmtx, tvecs), axis=1)
			

			# Funções de desenho na tela
			img = draw_axis(img,corners2,imgpts)
			img = draw_function(img,points2d.astype(int))
			cv2.imshow('Axis',img)
			times+=1

	k=cv2.waitKey(2)

	if k%256 == 27:
		break

cv2.destroyAllWindows()