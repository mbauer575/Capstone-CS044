#include <LiquidCrystal.h>
LiquidCrystal lcd(12,11,5,4,3,2);

const byte numChars=3;
char receivedChars[numChars];
bool newData=false;
int num_spots=0;
int num_cars=0;

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

    if(rc!='\n' && rc!='@'){
      receivedChars[ndx]=rc;
      ndx++;
      if(ndx>=numChars){
        ndx=numChars-1;
      }
    }else if(rc=='@'){
      receivedChars[ndx]='\0';
      ndx=0;
      newData=true;
      num_spots=atoi(receivedChars);
      Serial.read(); //Clear the \n out of the input buffer
    }else if(rc=='\n'){
      receivedChars[ndx]='\0';
      ndx=0;
      newData=true;
      num_cars=atoi(receivedChars);
    }
  }
}

void showNewData(){
  if(newData==true){
    lcd.clear();
    lcd.setCursor(0,0);
    int available=num_spots-num_cars;
    lcd.print(available);
    lcd.print(" spots open");
    lcd.setCursor(0,1);
    double percent;
    if(num_spots!=0){
      percent=((double)num_cars/(double)num_spots)*100;
      String convert=String(percent,1);
      Serial.print(percent);
      lcd.print(convert);
      lcd.print("% full");
    }
    newData=false;
  }
}