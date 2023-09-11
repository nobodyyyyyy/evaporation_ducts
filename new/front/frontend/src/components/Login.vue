<template>
<!--  <el-form class="login-container" label-position="left"-->
<!--           label-width="0px">-->
<!--    <h3 class="login_title">系统登录</h3>-->
<!--    <el-form-item>-->
<!--      <el-input type="text" v-model="loginForm.username"-->
<!--                auto-complete="off" placeholder="账号"></el-input>-->
<!--    </el-form-item>-->
<!--    <el-form-item>-->
<!--      <el-input type="password" v-model="loginForm.password"-->
<!--                auto-complete="off" placeholder="密码"></el-input>-->
<!--    </el-form-item>-->
<!--    <el-form-item style="width: 100%">-->
<!--      <el-button type="primary" style="width: 100%;background: #505458;border: none" v-on:click="login">登录</el-button>-->
<!--    </el-form-item>-->
<!--  </el-form>-->
  <div class="wrapper">
    <div class="background">
      <img :src="backgroundImg" width="100%" height="100%" alt="" />
    </div>
    <div class="content">
      <img class="title-img" :src="titleImg" alt=""/>
      <div class="entry">
        <el-button class="entry-btn" round v-on:click="login">进入系统</el-button>
      </div>
<!--      <el-container style="height: 100%;">-->
<!--        <el-aside width="400px">-->
<!--          <div class="title">-->
<!--            短临预测蒸发波导-->
<!--          </div>-->
<!--        </el-aside>-->
<!--        <el-main></el-main>-->
<!--      </el-container>-->
    </div>

<!--    <div class="title">-->
<!--      短临预测蒸发波导软件-->
<!--    </div>-->
  </div>
</template>

<script>

export default {
  name: 'Login',
  data () {
    return {
      backgroundImg: require('../assets/login.jpg'),
      titleImg: require('../assets/title.png'),
      loginForm: {
        username: '',
        password: ''
      },
      responseResult: []
    }
  },
  methods: {
    login () {
      // this.$axios
      //   .post('/api/user/info', {}, {
      //     // username: this.loginForm.username,
      //     // password: this.loginForm.password
      //     headers: {
      //       token: '666666'
      //     }
      //   })
      //   .then(successResponse => {
      //     if (successResponse.data.code === 200) {
      //       this.$router.replace({path: '/index'})
      //     } else {
      //       console.log(successResponse)
      //     }
      //   })
      //   .catch(failResponse => {
      //   })
      this.$axios.post('/login').then(res => {
        console.log(res.data)
        alert(res.data['name'])
      })
      const path = this.$route.query.redirect
      this.$router.replace({path: path === '/' || path === undefined ? '/Main' : path})
    }
  }
}
</script>

<style>

.background {
  width:100%;
  height:100%;  /**宽高100%是为了图片铺满屏幕 **/
  position: absolute;
  z-index:-1;
}

.content {
  width:100%;
  height:100%;  /**宽高100%是为了图片铺满屏幕 **/
  position: absolute;
  z-index:1;
}

.entry {
  //height: 20%;
  //width: 34%;
  //background: rgba(253, 247, 247, 0.24);
  //margin-top: 400px;
  //margin-left: 100px;
  border-radius: 20px;
  //text-align: left;
}

.entry-btn {
  width: 300px;
  height: 60px;
  font-size: 25px;
  border: 0;
  background: rgba(253, 252, 252, 0.8);
}

.title-img{
  margin-top: 70px;
  width: 60%;
  height: 20%;
}

.el-aside {
  background-color: rgba(243, 241, 241, 0.5);
  color: #333;
  //text-align: center;
  //line-height: 100px;
  height: 100%;
}

.el-main {
  //background-color: #E9EEF3;
  color: #333;
  text-align: center;
  line-height: 160px;
  height: 100%;
}

.title {
  color:black;
  text-align: left;
  font-size: 20px;
  margin-top: 250px;
  z-index: 1;
}

</style>
