class LiquidationCalculator:
    LEVERAGE_LEVELS = [2,3,5,10,20,25,50,75,100,125]
    @staticmethod
    def calculate_liquidation_levels(entry_price, position_type='long'):
        data=[]
        for lev in LiquidationCalculator.LEVERAGE_LEVELS:
            if position_type.lower()=='long':
                liq = entry_price * (1 - 0.95/lev); dist=(entry_price-liq)/entry_price*100
            else:
                liq = entry_price * (1 + 0.95/lev); dist=(liq-entry_price)/entry_price*100
            if dist>10: lvl, color='NIEDRIG','#28a745'
            elif dist>5: lvl,color='MITTEL','#ffc107'
            elif dist>2: lvl,color='HOCH','#fd7e14'
            else: lvl,color='EXTREM','#dc3545'
            data.append({'leverage':f"{lev}x",'liquidation_price':liq,'distance_percent':dist,'risk_level':lvl,'risk_color':color,'max_loss':100/lev})
        return data
