using Colors


TUred = colorant"rgb(141,45,57)"
TUdark = colorant"rgb(55,65,74)"
TUgold = colorant"rgb(174,159,109)"
TUlightgold = colorant"rgb(239,236,226)"
TUlightred = colorant"rgb(206,143,137)"
TUgray = colorant"rgb(175,179,183)"
TUsecondary1 = colorant"rgb(65,90,140)"
TUsecondary2 = colorant"rgb(0,105,170)"
TUsecondary3 = colorant"rgb(80,170,200)"
TUsecondary4 = colorant"rgb(130,185,160)"
TUsecondary5 = colorant"rgb(125,165,75)"
TUsecondary6 = colorant"rgb(50,110,30)"
TUsecondary7 = colorant"rgb(200,80,60)"
TUsecondary8 = colorant"rgb(175,110,150)"
TUsecondary9 = colorant"rgb(180,160,150)"
TUsecondary10 = colorant"rgb(215,180,105)"
TUsecondary11 = colorant"rgb(210,150,0)"
TUsecondary12 = colorant"rgb(145,105,70)"
lred = colorant"rgb(200,0,0)"
dred = colorant"rgb(130,0,0)"
dblu = colorant"rgb(0,0,130)"
dgre = colorant"rgb(0,130,0)"
dgra = colorant"rgb(50,50,50)"
mgra = colorant"rgb(221,222,214)"
lgra = colorant"rgb(238,238,234)"
MPG = colorant"rgb(0,125,122)"
lMPG = colorant"rgb(0,190,189)"
ora = colorant"#FF9933"
lblu = colorant"#7DA7D9"



gt_args = Dict(:color => :black, :lw => 3, :alpha => 0.9)
data_args = Dict(:color => TUsecondary4, :markersize => 5)
filter_estimate_args = Dict(:color => TUred, :lw => 4)
filter_cred_interval_args = Dict(:color => TUred, :ls => :dash, :lw => 2, :alpha => 0.4)
prediction_estimate_args = Dict(:color => dgra, :lw => 4)
prediction_cred_interval_args = Dict(:color => dgra, :ls => :dash, :lw => 2, :alpha => 0.4)
smoother_estimate_args = Dict(:color => colorant"rgb(19, 79, 209)", :lw => 4)
smoother_cred_interval_args = Dict(:color => colorant"rgb(19, 79, 209)", :ls => :dash, :lw => 2, :alpha => 0.4)

nothing