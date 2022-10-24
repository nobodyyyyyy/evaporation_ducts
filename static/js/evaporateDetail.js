// 文件选择相关
function detail(inputs) {
    var filepath = document.getElementById("file-data").value;
    console.log("文件路径: ", filepath);
    if (filepath == "") {
        document.getElementById("file-tips").innerHTML = "&nbsp;&nbsp;未选择文件，请先选择文件";
        document.getElementById("file-tips").style.color = "red";
        inputs.href = "#";
        return ;
    }
    var file = document.getElementById("file-data").files[0];
    var filetial =file.name.split('.');
    console.log(filetial);
    var filetial = filetial[filetial.length-1];
    if (filetial != "txt" && filetial != "tpu") {
        document.getElementById("file-tips").innerHTML = "&nbsp;&nbsp;文件格式有误，请选择txt或tpu文件";
        document.getElementById("file-tips").style.color = "red";
        document.getElementById("")
        inputs.href = "#";
    }
    else {
        inputs.href = "#infor-window";
        if (window.FileReader) {
            var reader = new FileReader();
            reader.onload = function () {
                showInDialog(this.result, inputs);
            }
            reader.readAsText(file);
        }
        else if (typeof window.ActiveXObject != 'undefined'){
            var xmlDoc;
            xmlDoc = new ActiveXObject("Microsoft.XMLDOM");
            xmlDoc.async = false;
            xmlDoc.load(filepath, inputs);
            showInDialog(xmlDoc.xml);
        }
        //支持FF
        else if (document.implementation && document.implementation.createDocument) {
            var xmlDoc;
            xmlDoc = document.implementation.createDocument("", "", null);
            xmlDoc.async = false;
            xmlDoc.load(filepath, inputs);
            showInDialog(xmlDoc.xml);
        } else {
            inputs.href = "#";
            alert('error');
        }
    }
}
function showInDialog(result) {
    document.getElementById("file-tips").innerHTML = "&nbsp;&nbsp;文件选择成功!";
    document.getElementById("file-tips").style.color = "#87CEFA";
    document.getElementById("file-infor").value = result;
}
// 异常提示框
window.alert =alert;
function alert(e) {
    $("body").append('<div id="msg"><div id="msg_top">错误信息<span class="msg_close">×</span></div><div id="msg_cont" style="color: red">'+e+'</div><div class="msg_close" id="msg_clear">确定</div></div>');
        $(".msg_close").click(function (){
            $("#msg").remove();
        });
}
function descriptionRow() {
    let status = document.getElementById("description-row").style.display;
    if (status == "none") {
        document.getElementById("description-row").style.display = "";
    } else {
        document.getElementById("description-row").style.display = "none";
    }
}

function descriptionRadar() {
    let status = document.getElementById("description-radar").style.display;
    if (status == "none") {
        document.getElementById("description-radar").style.display = "";
    } else {
        document.getElementById("description-radar").style.display = "none";
    }
}