#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// 서비스와 캐릭터리스틱 UUID는 임의로 지정 (PC와 동일하게 맞출 것)
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("Client connected");
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("Client disconnected");
  }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) override {
    std::string rxValue = pCharacteristic->getValue();

    if (rxValue.length() > 0) {
      Serial.print("Received via BLE: ");
      Serial.println(rxValue.c_str());
      // 필요시 받은 데이터 처리 가능
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(1000);

  BLEDevice::init("ESP32_BLE_Server");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE  |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pCharacteristic->setCallbacks(new MyCallbacks());
  pCharacteristic->setValue("Hello from ESP32");
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->start();

  Serial.println("BLE server started, advertising...");
}

void loop() {
  // 연결되었을 때만 notify 가능
  if (deviceConnected) {
    static unsigned long lastTime = 0;
    unsigned long now = millis();
    if (now - lastTime > 1000) {
      lastTime = now;

      // 여기서 원하는 데이터를 문자열로 세팅 후 notify
      std::string msg = "Data from ESP32";
      pCharacteristic->setValue(msg);
      pCharacteristic->notify();

      Serial.println("Notification sent: " + String(msg.c_str()));
    }
  }

  delay(20);
}
