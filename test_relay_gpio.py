#instalacion de GPIO en JetSon Nano:
#1) "git clone https://github.com/NVIDIA/jetson-gpio"
#2) "cd jetson-gpio"
#3) "sudo python3 setup.py install"
#4) "sudo groupadd -f -r gpio"
#5) Write "sudo usermod -a -G [username]" -> Example: sudo usermod -a -G gpio jetson
#6) "sudo cp lib/python/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/"
#7) "sudo udevadm control --reload-rules && sudo udevadm trigger"

import Jetson.GPIO as GPIO
import time

pinRelay_01 = 7
#pinRelay_02 = 5

GPIO.setmode(GPIO.BOARD)

GPIO.setup(pinRelay_01, GPIO.OUT)
outVal = GPIO.HIGH
GPIO.output(pinRelay_01, outVal)
while True:
	#if outVal == GPIO.HIGH:
	#	outVal = GPIO.LOW
	#else:
	#	outVal = GPIO.HIGH
	outVal ^= GPIO.HIGH
	print("LED "+str(outVal) )
	GPIO.output(pinRelay_01, outVal)
	time.sleep(1)

GPIO.cleanup()
