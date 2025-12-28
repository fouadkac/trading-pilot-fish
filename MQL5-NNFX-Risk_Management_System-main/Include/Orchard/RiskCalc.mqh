
//	Has a new bar been opened
bool NewBar( string symbol = NULL, int timeframe = 0, bool initToNow = false ) {

   datetime        currentBarTime  = iTime( symbol, ( ENUM_TIMEFRAMES )timeframe, 0 );
   static datetime previousBarTime = initToNow ? currentBarTime : 0;
   if ( previousBarTime == currentBarTime ) return ( false );
   previousBarTime = currentBarTime;
   return ( true );
}

double DoubleToTicks( string symbol, double value ) {
   return ( value / SymbolInfoDouble( symbol, SYMBOL_TRADE_TICK_SIZE ) );
}

double TicksToDouble( string symbol, double ticks ) {
   return ( ticks * SymbolInfoDouble( symbol, SYMBOL_TRADE_TICK_SIZE ) );
}

double PointsToDouble( string symbol, int points ) {
   return ( points * SymbolInfoDouble( symbol, SYMBOL_POINT ) );
}

double EquityPercent( double value ) {
   return ( AccountInfoDouble( ACCOUNT_EQUITY ) * value ); // Value is actually a decimal
}

double PercentSLSize( string symbol, double riskPercent,
                      double lots ) { // Risk percent is a decimal (1%=0.01)
   return ( RiskSLSize( symbol, EquityPercent( riskPercent ), lots ) );
}

double PercentRiskLots( string symbol, double riskPercent,
                        double slSize ) { // Risk percent is a decimal (1%=0.01)
   return ( RiskLots( symbol, EquityPercent( riskPercent ), slSize ) );
}

double RiskLots( string symbol, double riskAmount, double slSize ) { // Amount in account currency

   double ticks = DoubleToTicks( symbol, slSize );
   double tickValue =
      SymbolInfoDouble( symbol, SYMBOL_TRADE_TICK_VALUE ); // value of 1 tick for 1 lot
   double lotRisk  = ticks * tickValue;
   double riskLots = riskAmount / lotRisk;
   return ( riskLots );
}

double RiskSLSize( string symbol, double riskAmount, double lots ) { // Amount in account currency

   double tickValue =
      SymbolInfoDouble( symbol, SYMBOL_TRADE_TICK_VALUE ); // value of 1 tick for 1 lot
   double ticks  = riskAmount / ( lots * tickValue );
   double slSize = TicksToDouble( symbol, ticks );
   return ( slSize );
}
