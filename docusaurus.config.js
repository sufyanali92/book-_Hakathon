// @ts-check
import lightCodeTheme from 'prism-react-renderer/themes/github';
import darkCodeTheme from 'prism-react-renderer/themes/dracula';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI and Humanoid Robots',
  tagline: 'A simulation-first, hands-on textbook',
  favicon: 'img/favicon.ico',

  url: 'https://sufyanali92.github.io',
  baseUrl: '/book-_Hakathon/',
  organizationName: 'sufyanali92',
  projectName: 'book-_Hakathon',
  deploymentBranch: 'gh-pages',

  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/sufyanali92/book-_Hakathon/edit/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: '',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book',
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
      links: [],
      copyright: '© 2026 Sufyan Ali',
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['python', 'bash', 'json', 'cpp'],
    },
  },
};

export default config;
