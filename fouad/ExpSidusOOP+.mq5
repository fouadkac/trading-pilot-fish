
#include "CheckTrade.mqh"
#include "Trade.mqh"
#include "Sidus.mqh"

input double   Lot=1;          
input double spreadLevel=5.0;
input double StopLoss=0.055;
input double Profit=0.007;
input int numberBarOpenPosition=7;
input int numberBarStopPosition=2;

input double TrailingStop=0.002;

CheckTrade checkTrade;
Trade trade(StopLoss,Profit,Lot, TrailingStop);
Sidus sidus(numberBarOpenPosition,numberBarStopPosition);

bool flagStopLoss=false; 

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  { 
  return(checkTrade.OnCheckTradeInit(Lot));
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
    
if(!checkTrade.OnCheckTradeTick(Lot,spreadLevel)){
return;
} 
static datetime last_time;
datetime last_bar_time=(datetime)SeriesInfoInteger(Symbol(),Period(),SERIES_LASTBAR_DATE);  
if(last_time!=last_bar_time)
{
last_time=last_bar_time;     
}else{
return;
}

/*
static datetime last_time_daily;
datetime last_bar_time_daily=(datetime)SeriesInfoInteger(Symbol(),PERIOD_D1,SERIES_LASTBAR_DATE);  
if(last_time_daily!=last_bar_time_daily)
{
last_time_daily=last_bar_time_daily;
flagStopLoss=false;    
}

if(flagStopLoss==true)return;
*/

MqlRates mrate[];
ResetLastError();
if(CopyRates(Symbol(),Period(),0,numberBarStopPosition,mrate)<0)
     {
Print(GetLastError());
      return;
     }  
     
ArraySetAsSeries(mrate,true);

//-----------------------------------------------------------------------------

bool TradeSignalBuy=false;
bool TradeSignalSell=false;


TradeSignalBuy=sidus.OnTradeSignalBuy();
TradeSignalSell=sidus.OnTradeSignalSell();

bool TradeSignalBuyStop=false;
bool TradeSignalSellStop=false;

//TradeSignalBuyStop=sidus.OnTradeSignalBuyStop(mrate);
//TradeSignalSellStop=sidus.OnTradeSignalSellStop(mrate);

trade.Order(TradeSignalBuy,TradeSignalBuyStop,TradeSignalSell,TradeSignalSellStop);
trade.Trailing();

}
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
static int _deals;
ulong _ticket=0;

if(HistorySelect(0,TimeCurrent()))
  {
 int  i=HistoryDealsTotal()-1;

   if(_deals!=i) {    
   _deals=i; 
   } else { return; }

   if((_ticket=HistoryDealGetTicket(i))>0)
     {
      string _comment=HistoryDealGetString(_ticket,DEAL_COMMENT);      
      if(StringFind(_comment,"sl",0)!=-1) {       
      flagStopLoss=true;
      } 
        
     }
  }   
  }

//------------------------------------------------------------------------
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {   
 
  }    

