# importamos los paquetes necesarios
import cv2
import jetson.inference
import jetson.utils
from adafruit_servokit import ServoKit
import Jetson.GPIO as GPIO
# inicializamos configuracion y pocision inicial de los servos
kit = ServoKit(channels=16)
hombro = 90  # hombro servo angle
codo = 90  # codo servo angle
kit.servo[0].angle = hombro
kit.servo[1].angle = codo
GPIO.cleanup()
led_pin= 18
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW) 
import time 
# initialize the object detection model
#net = jetson.inference.detectNet("--model=prueba.onnx", "--labels=labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes")
net = jetson.inference.detectNet(argv=['--model=prueba.onnx', '--labels=labels.txt' , '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])  

# set display dimensions
dispW=640
dispH=480


# set the GPIO pin for the LED (use the correct pin number)



# inicializa la camara web
cam=cv2.VideoCapture(0)

while True:
    # hacemos captura de los frames
    ret, frame = cam.read()
    
    # Convierte la imagen a un formato compatible con el modelo
    image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_input = jetson.utils.cudaFromNumpy(image_input)
    
    # Ejecuta el modelo en la imagen
    detections = net.Detect(image_input, 640, 480)
    
    # recorremos las detecciones 
    for detection in detections:
        # sacamos la coordenadas del centro de la deteccion dentro del frame
        Xcentro = int(detection.Center[0])
        Ycentro = int(detection.Center[1])
        
        # sacamos el ancho y alto de la deteccion
        w = int(detection.Width)
        h = int(detection.Height)
        
        # Dibujamos un rectangulo
        cv2.rectangle(frame, (Xcentro-w//2, Ycentro-h//2), (Xcentro+w//2, Ycentro+h//2), (0, 0, 255), 2)
        GPIO.output(led_pin, GPIO.HIGH) 
        time.sleep(0.1)
        GPIO.output(led_pin, GPIO.LOW)
        # calculamos el error del hombro y codo
        errorHombro = Xcentro - dispW/2
        errorCodo = Ycentro - dispH/2
        
        # movemos el hombro y codo basados en el error
        if abs(errorHombro) > 15:
            hombro = hombro - errorHombro/50
        if abs(errorCodo) > 15:
            codo = codo - errorCodo/50
            
        # retringir hombro y codo a un rango maximo y minimo
        hombro = min(180, max(0, hombro))
        codo = min(180, max(0, codo))
        
        # establecemos angulos al servo
        kit.servo[0].angle = hombro
        kit.servo[1].angle = codo
    
    # mostramos la imagen 
    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)
    
    
    if cv2.waitKey(1) == ord('q'):
        break


#cerramos  todas las ventams
cam.release()
cv2.destroyAllWindows()
