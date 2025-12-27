//+------------------------------------------------------------------+
//|                                                        ExportOHLC|
//|                        Script to Export OHLC Data to CSV         |
//+------------------------------------------------------------------+
#property copyright "YourName"
#property version   "1.00"


// Input parameters
input string StartDate = "2024.12.01"; // Start date in YYYY.MM.DD format
input string EndDate   = "2025.01.01"; // End date in YYYY.MM.DD format
input string Symbol    = "XAUUSD";     // Symbol to export data for
input ENUM_TIMEFRAMES TimeFrame = PERIOD_M1; // Timeframe to export

//+------------------------------------------------------------------+
//| Main function                                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   // Convert input dates to datetime
   datetime start_date = StringToTime(StartDate);
   datetime end_date = StringToTime(EndDate);
   if(start_date == 0 || end_date == 0 || start_date >= end_date)
   {
      Print("Invalid date range. Please check StartDate and EndDate inputs.");
      return;
   }

   // File path
   string file_name = Symbol + "_OHLC.csv";
   int file_handle = FileOpen(file_name, FILE_CSV|FILE_WRITE| FILE_SHARE_WRITE, ";");
   if(file_handle < 0)
   {
      Print("Failed to create file: ", file_name);
      return;
   }

   // Write header
   FileWrite(file_handle, "Date", "Time", "Open", "High", "Low", "Close");

   // Retrieve data
   int total_bars = iBars(Symbol, TimeFrame);
   for(int i = total_bars - 1; i >= 0; i--)
   {
      datetime bar_time = iTime(Symbol, TimeFrame, i);
      if(bar_time < start_date || bar_time >= end_date)
         continue;

      double open  = iOpen(Symbol, TimeFrame, i);
      double high  = iHigh(Symbol, TimeFrame, i);
      double low   = iLow(Symbol, TimeFrame, i);
      double close = iClose(Symbol, TimeFrame, i);

      // Write data to file
      FileWrite(file_handle, TimeToString(bar_time, TIME_DATE), TimeToString(bar_time,TIME_MINUTES), open, high, low, close);
   }

   // Close the file
   FileClose(file_handle);
   Print("Data successfully exported to ", file_name);
}
