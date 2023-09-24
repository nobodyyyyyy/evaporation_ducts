<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  font-family: ”Microsoft YaHei”,Arial,Helvetica,sans-serif,”宋体”;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
  height: 100%;
  z-index: -1;
}

html, body, #app{
  font-family: ”Microsoft YaHei”,Arial,Helvetica,sans-serif,”宋体”;
  margin: 0;
  padding: 0;
  height: 100%;
}

</style>
