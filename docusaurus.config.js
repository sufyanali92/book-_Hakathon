// @ts-check
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics – AI Systems in the Physical World',
  tagline: 'AI-Native Technical Textbook Platform',
  favicon: 'img/favicon.ico',

  // Vercel deployment config (serve from root)
  url: 'https://physical-ai-text-book.vercel.app',
  baseUrl: '/',

  // Still fine to keep for repo metadata
  organizationName: 'sufyanali92',
  projectName: 'book3',

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
          editUrl: 'https://github.com/GOV-STU/book2/tree/main/',
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
      title: '',
      logo: undefined,
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          to: '/chatbot',
          label: 'Chatbot',
          position: 'left',
        },
        {
          href: 'https://github.com/sufyanali92/book-_Hakathon',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: undefined,

    prism: {
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
    },
  },

  baseUrlIssueBanner: true,
};

module.exports = config;




// // @ts-check
// /** @type {import('@docusaurus/types').Config} */
// const config = {
//   title: 'Physical AI & Humanoid Robotics – AI Systems in the Physical World',
//   tagline: 'AI-Native Technical Textbook Platform',
//   favicon: 'img/favicon.ico',

//   // Vercel deployment config (serve from root)
//   url: 'https://physical-ai-text-book.vercel.app',
//   baseUrl: '/',

//   // Still fine to keep for repo metadata
//   organizationName: 'GOV-STU',
//   projectName: 'book2',

//   markdown: {
//     hooks: {
//       onBrokenMarkdownLinks: 'warn',
//       onBrokenMarkdownImages: 'ignore',
//     },
//   },

//   i18n: {
//     defaultLocale: 'en',
//     locales: ['en'],
//   },

//   presets: [
//     [
//       'classic',
//       {
//         docs: {
//           sidebarPath: require.resolve('./sidebars.js'),
//           editUrl: 'https://github.com/GOV-STU/book2/tree/main/',
//           routeBasePath: '/', // Docs at site root
//         },
//         blog: false,
//         theme: {
//           customCss: require.resolve('./src/css/custom.css'),
//         },
//       },
//     ],
//   ],

//   themeConfig: {
//     image: undefined,

//     navbar: {
//       title: '',
//       logo: undefined,
//       items: [
//         {
//           type: 'docSidebar',
//           sidebarId: 'tutorialSidebar',
//           position: 'left',
//           label: 'Textbook',
//         },
//         {
//           to: '/chatbot',
//           label: 'Chatbot',
//           position: 'left',
//         },
//         {
//           href: 'https://github.com/GOV-STU/book2',
//           label: 'GitHub',
//           position: 'right',
//         },
//       ],
//     },

//     footer: undefined,

//     prism: {
//       theme: require('prism-react-renderer').themes.github,
//       darkTheme: require('prism-react-renderer').themes.dracula,
//     },
//   },

//   baseUrlIssueBanner: true,
// };

// module.exports = config;
