import sys
from time import sleep
from gpiozero import LED, Button

import config


class HardwareError(Exception):
    pass

# TODO: モーターとポテンショメータの配線テストを追加
class HardwareTest:
    def __init__(self):
        self.leds = {
            'LED1': LED(config.LED_STATUS),
            'LED2': LED(config.LED_TASK_1_3),
            'LED3': LED(config.LED_TASK_4_6),
        }
        self.buttons = {
            'Button1': Button(config.BUTTON_START, pull_up=True),
            'Button2': Button(config.BUTTON_STOP, pull_up=True),
        }
    
    def test_led(self, led_name):
        if led_name not in self.leds:
            raise HardwareError(f"LED: {led_name} dose not exist.")
        
        self.leds[led_name].on()
        sleep(1)
        self.leds[led_name].off()
        print(f"{led_name}: OK")
    
    def test_all_leds(self):
        for led_name in self.leds:
            self.test_led(led_name)

    def test_button(self, button_name, timeout=5):
        if button_name not in self.buttons:
            raise HardwareError(f"Button '{button_name}' does not exist.")
        
        print(f"Press {button_name} (within {timeout}s)...")
        button = self.buttons[button_name]

        if button.wait_for_press(timeout=timeout):
            print(f"{button_name}: OK")
        else:
            raise HardwareError(f"{button_name}: Test failed (timeout)")
    
    def test_all_buttons(self):
        for button_name in self.buttons:
            self.test_button(button_name)

    def cleanup(self):
        for led in self.leds.values():
            led.off()
        print("Cleanup complete!")


if __name__ == "__main__":
    test = HardwareTest()

    try:
        # test.test_led('LED1')
        test.test_all_leds()
        test.test_all_buttons()
    
    except HardwareError as e:
        print(f"\nHardware test failed: {e}")
        sys.exit(1)


    finally:
        test.cleanup()