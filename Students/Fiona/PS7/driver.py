from PS7 import demographics as demog

#population parameters
E= 20
S =80
tot_per= E+S
beg_yr_demog = 1
end_yr_demog = 100
fert_graph = True
mort_graph = True
imm_graph = True
fert_rates = demog.get_fer(tot_per, beg_yr_demog, end_yr_demog, fert_graph)
mort_rates = demog.ger_mort(tot_per, beg_yr_demog, end_yr_demog, mort_graph)
imm_rates = demog.ger_imm_resid(tot_per, beg_yr_demog, end_yr_demog, imm_graph)

