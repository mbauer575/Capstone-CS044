#include <LiquidCrystal.h>
LiquidCrystal lcd(12,11,5,4,3,2);

const byte numChars=3;
char receivedChars[numChars];
bool newData=false;

void setup() {
  Serial.begin(9600);

  lcd.begin(16,2);
  lcd.print("Start");
}

void loop() {
  recv();
  showNewData();
}

void recv(){
  static byte ndx=0;
  char rc;

  while(Serial.available()>0 && newData==false){
    rc=Serial.read();

    if(rc!='\n'){
      receivedChars[ndx]=rc;
      ndx++;
      if(ndx>=numChars){
        ndx=numChars-1;
      }
    }else{
      receivedChars[ndx]='\0';
      ndx=0;
      newData=true;
    }
  }
}

void showNewData(){
  if(newData==true){
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Spots available");
    lcd.setCursor(0,1);
    lcd.print(receivedChars);
    newData=false;
  }
}