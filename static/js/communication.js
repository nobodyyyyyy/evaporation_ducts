$(document).ready(function() {
    namespace = '/test_conn';
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace, {transports: ['websocket']});
    console.log(socket)
    socket.on('server_response', function(res) {
        var basicdata = res.data;
        console.log(basicdata)
        setoption(basicdata)
    });
});