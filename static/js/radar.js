// $(document).ready(function() {
//     namespace = '/test_conn';
//     var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace, {transports: ['websocket']});
//     socket.on('server_response', function(res) {
//         var basicdata = res.data;
//         setoption(basicdata)
//     });
// });

function radar_detail(){
    var btn = document.getElementById("detail-button").innerHTML;
    console.log(123);
    if(btn=="隐藏"){
        document.getElementById("radar-infor").style.display="none";
        document.getElementById("detail-button").innerHTML="详细";
    }
    else{
        document.getElementById("radar-infor").style.display="";
        document.getElementById("detail-button").innerHTML="隐藏";
    }
}

function radar_infor(){
    var RF = document.getElementById("radar-fre").value;
    var RT = document.getElementById("radar-top").value;
    var AH = document.getElementById("antenna-high").value;
    var AG = document.getElementById("antenna-gain").value;
    var BW = document.getElementById("beam-width").value;
    var LE = document.getElementById("launch-ele").value;
    var MN = document.getElementById("min-noise").value;
    var RW = document.getElementById("rec-width").value;
    var SL = document.getElementById("sys-loss").value;
    var NC = document.getElementById("noise-coe").value;
    var TH = document.getElementById("target-high").value;
    var RR = document.getElementById("rcs-radar").value;
    var R_I = "请确认以下输入信息：\n雷达频率："+RF+"   雷达峰值频率："+RT+
    "\n天线高度："+AH+"   天线增益："+AG+"\n波束宽度："+BW+"   发射仰角："+LE+
    "\n最小信噪比："+MN+"   接收机带宽："+RW+"\n系统综合损耗："+SL+"   接收机噪声系数："+
    NC+"\n目标高度："+TH+"   目标散射界面"+RR
    return confirm(R_I)
}