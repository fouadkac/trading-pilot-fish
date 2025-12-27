#include <Files\FileTxt.mqh>

// Structure to define export configuration
struct ExportConfig
{
   ENUM_TIMEFRAMES tf;
   int bars;
   string tf_label;
};

// Define multiple timeframes and bar counts
ExportConfig configs[] = {
   { PERIOD_H4,  5000, "H4"   },
   { PERIOD_H1,  20000, "H1"   },
   { PERIOD_M15, 80000, "M15"  },
   { PERIOD_M5,  240000, "M5"   },
   { PERIOD_M1,  2000000, "M3"   } // 3-minute may need custom data or synthetic generation
};

void OnStart()
{
   string symbol = _Symbol;

   for(int cfg = 0; cfg < ArraySize(configs); cfg++)
   {
      ENUM_TIMEFRAMES tf = configs[cfg].tf;
      int bars = configs[cfg].bars;
      string tf_label = configs[cfg].tf_label;

      string filename = symbol + "_TF" + tf_label + ".csv";
      int file = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI);
      
      if(file == INVALID_HANDLE)
      {
         Print("Error opening file ", filename, ": ", GetLastError());
         continue;
      }

      FileWrite(file, "Time", "Open", "High", "Low", "Close", "Volume");

      for(int i = bars - 1; i >= 0; i--)
      {
         datetime time = iTime(symbol, tf, i);
         if(time == 0) continue; // Skip missing bars

         double open = iOpen(symbol, tf, i);
         double high = iHigh(symbol, tf, i);
         double low = iLow(symbol, tf, i);
         double close = iClose(symbol, tf, i);
         long vol = iVolume(symbol, tf, i);

         FileWrite(file,
                   TimeToString(time, TIME_DATE | TIME_MINUTES),
                   DoubleToString(open, 5),
                   DoubleToString(high, 5),
                   DoubleToString(low, 5),
                   DoubleToString(close, 5),
                   (int)vol);
      }

      FileClose(file);
      Print("Exported ", bars, " bars from ", tf_label, " to ", filename);
   }
}
