/*
 * 完整项目: ESP32S3 蓝牙步进电机调音器从控 (TMC2209 优化版)
 * 目标: 利用 TMC2209 解决 TMC2208 的低保持电流限制问题。
 * 硬件: Seeed Studio XIAO ESP32S3, TMC2209 驱动器。
 * 依赖库: 
 * - BLEDevice (ESP32 内核自带)
 * - TMCStepper
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <TMCStepper.h> 
#include "driver/ledc.h" 

// --- 函数声明 ---
void moveMotor(int steps);
void processCommand(String command);
void motorControlTask(void * parameter);

// --- 驱动器类：兼容TMC2209和TMC2208的寄存器访问 ---
class MyTMC2209 : public TMC2209Stepper {
public:
    MyTMC2209(Stream * SerialPort, float RS, uint8_t addr) : 
        TMC2209Stepper(SerialPort, RS, addr) {}
};

// --- BLE 定义 ---
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define DEVICE_NAME         "ESP32_Tuner"

// --- TMC2209 & 电机引脚 (XIAO S3 引脚定义，与2208相同) ---
#define TMC_UART_RX_PIN 5  
#define TMC_UART_TX_PIN 6  
#define PUL_PIN   7      
#define DIR_PIN   8      
#define ENA_PIN   9      

// --- 电机参数 & PWM 定义 ---
#define R_SENSE 0.11f         
#define DRIVER_ADDRESS 0
#define MICROSTEPS 16         
#define STEPS_PER_CENT 10     
#define MOTOR_SPEED 800       

// PWM/LEDC 配置
#define LEDC_SPEED_MODE LEDC_LOW_SPEED_MODE 
#define LEDC_TIMER      LEDC_TIMER_0        
#define LEDC_CHANNEL    LEDC_CHANNEL_0      
#define PULSE_FREQUENCY MOTOR_SPEED         

#define LEDC_RESOLUTION LEDC_TIMER_10_BIT 
#define DUTY_CYCLE      (1 << 10) / 2    

// ** 关键扭矩优化 **
#define MOTOR_RUN_CURRENT 1400 // 设定运行电流为 1.4A (1400mA)

// --- 变量 ---
HardwareSerial& motorSerial = Serial2; 
// *** 关键修改：使用TMC2209Stepper类 ***
MyTMC2209 driver(&motorSerial, R_SENSE, DRIVER_ADDRESS); 
bool deviceConnected = false;
volatile int stepsRemaining = 0; 
portMUX_TYPE stepsRemainingMux = portMUX_INITIALIZER_UNLOCKED; 
BLECharacteristic *pCharacteristic;

// --- BLE 回调类 ---
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Device Connected");
    };
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Device Disconnected");
      BLEDevice::startAdvertising();
    }
};

class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      uint8_t* data = pCharacteristic->getData();
      size_t length = pCharacteristic->getLength(); 
      if (length > 0) {
        String cmdStr;
        for (int i = 0; i < length; i++) {
            cmdStr += (char)data[i];
        }
        processCommand(cmdStr);
      }
    }
}; 

// --- FreeRTOS 任务：在 Core 1 上计时和停止电机 ---
void motorControlTask(void * parameter) {
    for (;;) {
        portENTER_CRITICAL(&stepsRemainingMux);
        int currentSteps = stepsRemaining; 
        portEXIT_CRITICAL(&stepsRemainingMux);

        if (currentSteps > 0) {
            int total_run_time_ms = (currentSteps * 1000) / MOTOR_SPEED;
            
            vTaskDelay(total_run_time_ms / portTICK_PERIOD_MS);
            
            // 1. 停止 PWM 
            ledc_set_duty(LEDC_SPEED_MODE, LEDC_CHANNEL, 0); 
            ledc_update_duty(LEDC_SPEED_MODE, LEDC_CHANNEL); 
            
            // 2. *** TMC2209优化：移除多余的ENA刷新逻辑 ***
            // TMC2209会尊重ihold(31)的设置，无需额外刷新。
            
            portENTER_CRITICAL(&stepsRemainingMux);
            stepsRemaining = 0; 
            portEXIT_CRITICAL(&stepsRemainingMux);
            
            Serial.println("Motor move complete. TMC2209 should maintain high torque.");
        }
        // 当没有运动时，任务休眠 10ms
        vTaskDelay(10 / portTICK_PERIOD_MS); 
    }
}

// --- setup 函数 ---
void setup() {
  Serial.begin(115200);
  
  // 1. 电机引脚配置
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);
  digitalWrite(ENA_PIN, LOW); // 默认使能驱动器
  
  // 2. TMC2209 初始化
  motorSerial.begin(115200, SERIAL_8N1, TMC_UART_RX_PIN, TMC_UART_TX_PIN);
  driver.begin(); 
  driver.microsteps(MICROSTEPS); 
  
  // ********* TMC2209 关键电流和模式配置 *********
  
  // 1. 启用 SpreadCycle (性能模式)
  // 此操作会禁用静音模式，确保高扭矩。
  driver.en_spreadCycle(true);
  
  // 2. 设定运行电流 (RMS Current) 为 1.4A
  driver.rms_current(MOTOR_RUN_CURRENT); 
  
  // 3. 设定保持电流 (Hold Current) 为运行电流的 97%
  // TMC2209会尊重此值。
  driver.ihold(31); 
  
  // 4. 禁用自动电流降低延时
  driver.TPOWERDOWN(255); 
  // **********************************
  
  driver.pwm_autograd(true);
  driver.pdn_disable(true); 
  Serial.println("TMC2209 Initialization attempt complete.");
  Serial.print("TMC2209 GCONF value: 0x");
  Serial.println(driver.GCONF(), HEX); 
  
  // 3. LEDC PWM 配置
  ledc_timer_config_t ledc_timer = {
      .speed_mode = LEDC_SPEED_MODE,    
      .duty_resolution = LEDC_RESOLUTION,   
      .timer_num = LEDC_TIMER,              
      .freq_hz = PULSE_FREQUENCY,           
      .clk_cfg = LEDC_AUTO_CLK,             
  };
  ledc_timer_config(&ledc_timer);

  ledc_channel_config_t ledc_channel = {
      .gpio_num   = PUL_PIN,                
      .speed_mode = LEDC_SPEED_MODE,    
      .channel    = LEDC_CHANNEL,           
      .intr_type  = LEDC_INTR_DISABLE,      
      .timer_sel  = LEDC_TIMER,             
      .duty       = 0,                      
      .hpoint     = 0,                      
  };
  ledc_channel_config(&ledc_channel);
  
  // 4. BLE 初始化
  BLEDevice::init(DEVICE_NAME);
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks()); 
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_WRITE
                    );
  pCharacteristic->setCallbacks(new MyCallbacks()); 
  
  // 添加 BLE2902 描述符
  pCharacteristic->addDescriptor(new BLE2902());
  
  pService->start();
  
  // 5. 启动 BLE 广告
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06); 
  BLEDevice::startAdvertising();
  Serial.println("Waiting for Kivy App to connect...");
  
  // 6. 启动 FreeRTOS 任务 (绑定到 Core 1)
  xTaskCreatePinnedToCore(
      motorControlTask,   
      "MotorCtrl",        
      4096,               
      NULL,               
      1,                  
      NULL,               
      1                   // 绑定到 Core 1
  );
}

// --- loop 函数 ---
void loop() {
  delay(10); 
}

// --- Non-blocking 版本的 moveMotor ---
void moveMotor(int steps) {
  portENTER_CRITICAL(&stepsRemainingMux);
  if (stepsRemaining > 0) {
    portEXIT_CRITICAL(&stepsRemainingMux);
    Serial.println("Motor is busy. Command ignored.");
    return;
  }
  stepsRemaining = steps; 
  portEXIT_CRITICAL(&stepsRemainingMux);
  
  // 启动 PWM 脉冲输出 
  ledc_set_duty(LEDC_SPEED_MODE, LEDC_CHANNEL, DUTY_CYCLE); 
  ledc_update_duty(LEDC_SPEED_MODE, LEDC_CHANNEL); 
  
  Serial.printf("Starting non-blocking motor move for %d steps.\n", steps);
}

// --- 命令解析函数 ---
void processCommand(String command) {
  int commaIndex = command.indexOf(',');
  if (commaIndex == -1) return;
  
  String action = command.substring(0, commaIndex);
  String valStr = command.substring(commaIndex + 1);
  float cents = valStr.toFloat();
  
  int stepsToMove = (int)(cents * STEPS_PER_CENT);
  if (stepsToMove < 5) stepsToMove = 5; 
  
  if (action == "STEP_UP") {
    Serial.printf("Tuning UP (Sharpening): %d steps\n", stepsToMove);
    digitalWrite(DIR_PIN, HIGH); 
    moveMotor(stepsToMove); 
  } 
  else if (action == "STEP_DOWN") {
    Serial.printf("Tuning DOWN (Flattening): %d steps\n", stepsToMove);
    digitalWrite(DIR_PIN, LOW); 
    moveMotor(stepsToMove);
  }
}