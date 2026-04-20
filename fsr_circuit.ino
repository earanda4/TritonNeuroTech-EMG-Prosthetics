#include <LiquidCrystal.h>

// RS, E, D4, D5, D6, D7
LiquidCrystal lcd(7, 8, 9, 10, 11, 12);

void setup() {
  Serial.begin(9600);

  lcd.begin(16, 2);
  lcd.print("FSR System");
  delay(1000);
  lcd.clear();
}

void loop() {
  int fsr1 = analogRead(A0);
  int fsr2 = analogRead(A1);
  int fsr3 = analogRead(A2);
  int fsr4 = analogRead(A3);

  // Serial output (easier to debug)
  Serial.print("1:");
  Serial.print(fsr1);
  Serial.print(" 2:");
  Serial.print(fsr2);
  Serial.print(" 3:");
  Serial.print(fsr3);
  Serial.print(" 4:");
  Serial.println(fsr4);

  // LCD display (2 lines max)

  lcd.setCursor(0, 0);
  lcd.print("1:");
  lcd.print(fsr1);
  lcd.print(" 2:");
  lcd.print(fsr2);
  lcd.print("   "); // clears leftover digits

  lcd.setCursor(0, 1);
  lcd.print("3:");
  lcd.print(fsr3);
  lcd.print(" 4:");
  lcd.print(fsr4);
  lcd.print("   ");

  delay(200);
}
