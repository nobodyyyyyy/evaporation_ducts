<template>
  <div @touchmove.prevent @mousewheel.prevent id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-main>
        1
        <!--<books></books>-->
      </el-main>
    </el-container>
  </div>
</template>

<script>

import SideMenu from './SideMenu'
export default {
  name: 'AppLibrary',
  components: {SideMenu},
  methods: {
    listByCategory () {
      // eslint-disable-next-line no-unused-vars
      var _this = this
      // eslint-disable-next-line no-unused-vars
      var cid = this.$refs.sideMenu.cid
      console.log(cid)
      // var url = 'categories/' + cid + '/books'
      // this.$axios.get(url).then(resp => {
      //   if (resp && resp.status === 200) {
      //     _this.$refs.booksArea.books = resp.data
      //   }
      // })
    }
  }
}
</script>

<style scoped>
#wrapper {
  height: 95%;
}

.el-header, .el-footer {
  background-color: #B3C0D1;
  color: #333;
  text-align: center;
  line-height: 60px;
}

.el-aside {
  //background-color: #D3DCE6;
  background-color: #E9EEF3;
  color: #333;
  text-align: center;
  line-height: 200px;
}

.el-main {
  background-color: #E9EEF3;
  color: #333;
  text-align: center;
  line-height: 160px;
}

body > .el-container {
  margin-bottom: 40px;
}

.el-container:nth-child(5) .el-aside,
.el-container:nth-child(6) .el-aside {
  line-height: 260px;
}

.el-container:nth-child(7) .el-aside {
  line-height: 320px;
}

</style>
