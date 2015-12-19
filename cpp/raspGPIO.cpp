//
//  raspGPIO: Raspberry Pi GPIO module
//      copy from http://www.myu.ac.jp/~xkozima/lab/raspTutorial3.html
#include "raspGPIO.h"
#include <stdio.h>

#define PERI_BASE     0x3F000000 //for rasberry pi 2
#define GPIO_BASE     (PERI_BASE + 0x200000)
#define BLOCK_SIZE    4096

static volatile unsigned int *Gpio;

void gpio_init()
{
    if (Gpio) return;
    int fd;
    void *gpio_map;
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd == -1) {
        printf("error: cannot open /dev/mem (gpio_init)\n");
        exit(-1);
    }
    gpio_map = mmap(NULL, BLOCK_SIZE,
                    PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd, GPIO_BASE );
    if ((int) gpio_map == -1) {
        printf("error: cannot map /dev/mem on the memory (gpio_init)\n");
        exit(-1);
    }
    close(fd);
    Gpio = (unsigned int *) gpio_map;
}

//      pin : (P1) 2,3,4,7,8,9,10,11,14,15,17,18,22,23,24,25,27
//            (P5) 28,29,30,31
//      mode: GPIO_INPUT, _OUTPUT, _ALT0, _ALT1, _ALT2, _ALT3, _ALT4, _ALT5
void gpio_configure(int pin, int mode)
{
    if (pin < 0 || pin > 31) {
        printf("error: pin number out of range (gpio_configure)\n");
        exit(-1);
    }
    int index = pin / 10;
    unsigned int mask = ~(0x7 << ((pin % 10) * 3));
    Gpio[index] = (Gpio[index] & mask) | ((mode & 0x7) << ((pin % 10) * 3));
}

void gpio_set(int pin)
{
    if (pin < 0 || pin > 31) {
        printf("error: pin number out of range (gpio_set)\n");
        exit(-1);
    }
    Gpio[7] = 0x1 << pin;   //  GPSET0
}
void gpio_clear(int pin)
{
    if (pin < 0 || pin > 31) {
        printf("error: pin number out of range (gpio_clear)\n");
        exit(-1);
    }
    Gpio[10] = 0x1 << pin;  //  GPCLR0
}

int gpio_read (int pin)
{
    if (pin < 0 || pin > 31) {
        printf("error: pin number out of range (gpio_read)\n");
        exit(-1);
    }
    return (Gpio[13] & (0x1 << pin)) != 0;  //  GPLEV0
}

void gpio_configure_pull (int pin, int pullmode)
{
    if (pin < 0 || pin > 31) {
        printf("error: pin number out of range (gpio_configure_pull)\n");
        exit(-1);
    }
    Gpio[37] = pullmode & 0x3;  //  GPPUD
    usleep(1);
    Gpio[38] = 0x1 << pin;      //  GPPUDCLK0
    usleep(1);

    Gpio[37] = 0;
    Gpio[38] = 0;
}

int main(int argc, char *argv[])
{
    int clocking_pin = 5;
    gpio_init();
    if(argc == 1){
        printf("End clocking\n");
        gpio_configure(5, GPIO_INPUT);  
    }
    else{
        gpio_init();
        if(argv[1][0] == '5'){
             printf("Start clocking\n");
             gpio_configure_pull(clocking_pin, GPIO_PULLDOWN);
             gpio_configure(clocking_pin, GPIO_ALT0);
        }else if(argv[1][0] == '4'){
             printf("Start clocking\n");
             gpio_configure(clocking_pin, GPIO_ALT0);
        }
    }
    return 0;
}

//