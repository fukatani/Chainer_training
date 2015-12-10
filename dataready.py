import RPi.GPIO as GPIO
import time

GPIO_DR = 5

def callback(pin):
    time.sleep(0.001)
    #do spi

GPIO.setup(BCM)
GPIO.setup(GPIO_DR, GPIO.IN, pull_up_down=GPIO.PUD_UP)


GPIO.add_event_detect(GPIO_DR, GPIO.RISING)
GPIO.add_event_callback(GPIO_DR, callback)


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
