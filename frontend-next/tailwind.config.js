/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        joy: '#FFD700',
        sad: '#4169E1',
        anxiety: '#9370DB',
        anger: '#DC143C',
        neutral: '#808080',
      },
    },
  },
  plugins: [],
}
