// @ts-check
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics – AI Systems in the Physical World',
  tagline: 'AI-Native Technical Textbook Platform',
  favicon: 'img/favicon.ico',

  // ✅ Vercel root deployment
  url: 'https://book-hakathon.vercel.app',
  baseUrl: '/',

  organizationName: 'sufyanali92',
  projectName: 'book-_Hakathon',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'ignore',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/sufyanali92/book-_Hakathon/tree/main/',
          routeBasePath: '/', // Docs at site root
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    image: undefined,

    navbar: {
      title: 'Book',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/sufyanali92/book-_Hakathon',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      copyright: `© ${new Date().getFullYear()} Sufyan Ali. All rights reserved.`,
    },

    // ✅ FIXED Prism import (Docusaurus v3)
    prism: {
       theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
    },
  },

  baseUrlIssueBanner: true,
};

module.exports = config;





