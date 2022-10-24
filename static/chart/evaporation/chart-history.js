var his_data = []
var his_time = []
var his_latlon = []
function show_his(){
    // $("#his_btn").click(function(){
    //     $("#history_form").attr("target","iframe");
    // });
    if(Requ==null){
        //document.getElementById("time_select").innerHTML = "暂无该时段蒸发波导信息，请重新选择区间查询："
        return
    }
    var data_num = count(Requ);
    if(data_num == 0){
        document.getElementById("time_select").innerHTML = "暂无该时段蒸发波导信息，请重新选择区间查询："
        return
    }
    document.getElementById("chart-history").style.display = "";
    document.getElementById("research").style.display = "";
    document.getElementById("search-his").style.display = "none";
    var chart_history = echarts.init(document.getElementById('chart-history'))
    his_data = []
    his_time = []
    his_latlon = []
    var data_num = count(Requ);
    console.log(data_num)
    for(i=0;i<data_num;i++){
        his_data.push(Requ[i].evap)
        his_time.push(Requ[i].times)
        his_latlon.push(Requ[i].lat_lon)
    }
option_his = {
    tooltip: {
        trigger: 'axis',
        formatter: function (params) {
            params = params[0]
            return (
                '地理位置：' + his_latlon[params.dataIndex] + 
                '<br />时间： ' + his_time[params.dataIndex] + 
                '<br />波导高度： ' + his_data[params.dataIndex]
            );
        },
    },
    xAxis: {
        type: 'category',
        boundaryGap: false,
        axisLine: {onZero: true},
        data: his_time
    },
    yAxis: {
        splitLine: {
            show: false
        }
    },
    toolbox: {
        left: 'center',
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            dataView: {
                readOnly: true, 
                optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>蒸发波导</td><td>地理位置</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + his_time[i] + '</td>' + 
                                 '<td>' + his_data[i] + '</td>' + 
                                 '<td>' + his_latlon[i] + '</td>' +
                                 '</tr>';
                    }
                    table += '</tbody></table>';
                    return table;
                }
            },
            magicType: {type: ['line', 'bar']},
            restore: {},
            saveAsImage: {}
        }
    },
    dataZoom: [
        {
            show: true,
            realtime: true,
            start: 50,
            end: 100,
        }
    ],
    visualMap: {
        show: false,
        type: 'continuous',
        min: 0,
        max: 30
    },
    series: {
        name: 'evaporation wave',
        type: 'line',
        data: his_data,
    }
};
    chart_history.setOption(option_his)
    p = document.getElementById("contentscroll2").innerHTML;
    if (RequestTime != null) {
        p +="\n 选择时间区间为"+ RequestTime.start_time + " -- " + RequestTime.end_time + "的蒸发波导高度历史信息查询...";
    }
    document.getElementById("contentscroll2").innerHTML = p;
}
function hidden_his(){
    document.getElementById("chart-history").style.display = "none";
    document.getElementById("research").style.display = "none";
    document.getElementById("search-his").style.display = "";
}
show_his()