const path = require('path');

module.exports = {
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src/'),
      '@components': path.resolve(__dirname, 'src/components/'),
      '@components/ui': path.resolve(__dirname, 'src/components/ui/')
    },
    extensions: ['.js', '.jsx', '.ts', '.tsx']
  }
};
