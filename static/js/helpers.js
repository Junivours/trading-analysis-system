// Shared JS helpers extracted from inline script
window.TradingHelpers = (function(){
  function signalColor(signal){
    if(!signal) return '#6c757d';
    const s = signal.toString().toLowerCase();
    if(s.includes('buy') || s.includes('bull')) return '#28a745';
    if(s.includes('sell') || s.includes('bear')) return '#dc3545';
    return '#6c757d';
  }
  function rsiColor(rsi){
    if(rsi == null) return '#17a2b8';
    if(rsi > 80) return '#dc3545';
    if(rsi < 20) return '#28a745';
    if(rsi > 70) return '#fd7e14';
    if(rsi < 30) return '#20c997';
    return '#17a2b8';
  }
  function formatVolume(v){
    const num=parseFloat(v); if(!isFinite(num)) return '-';
    if(num>=1e9) return (num/1e9).toFixed(1)+'B';
    if(num>=1e6) return (num/1e6).toFixed(1)+'M';
    if(num>=1e3) return (num/1e3).toFixed(1)+'K';
    return num.toFixed(0);
  }
  return { signalColor, rsiColor, formatVolume };
})();
