import spidev
import time
import RPi.GPIO as GPIO

def callback(pin):
    time.sleep(0.00001)
    _, resp_upper = spi.xfer2([0x83,0x00])
    _, resp_lower = spi.xfer2([0x82,0x00])
    resp = resp_upper * 2 ** 8 + resp_lower
    print(resp)
    write_file.write(str(resp) + '\n')

def int_communication(debug=True):
    GPIO_DR = 6
    write_file = open('data.txt', 'w')

    spi = spidev.SpiDev()
    spi.open(0,0)
    spi.max_speed_hz=(1000000)
    if debug: #for no dataready
        callback(6)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_DR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(GPIO_DR, GPIO.RISING)
    GPIO.add_event_callback(GPIO_DR, callback)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
        write_file.close()

if __name__ == '__main__':
    int_communication(False)
